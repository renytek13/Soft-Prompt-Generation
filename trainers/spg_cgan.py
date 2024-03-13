import datetime
import pickle
import random
import time
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from tqdm import *
from utils.clip_part import load_clip_to_cpu, BaseTextEncoder
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import dassl
from dassl.utils.meters import AverageMeter, MetricMeter
from dassl.utils.tools import mkdir_if_missing
from dassl.utils.torchtools import partial
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler


_tokenizer = _Tokenizer()



class Generator(nn.Module):
    def __init__(self, cfg, n_ctx):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(cfg.FEAT_DIM + cfg.LATENT_DIM, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, n_ctx * 512),
            nn.Tanh()
        )

        # Initialization with He
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')


    def forward(self, image, n_ctx):
        # generate random noise
        noise = torch.randn([image.shape[0], 100]).to(image.device)
        noise = noise.view(noise.shape[0], -1)
        # concatenate noise and image label to produce gen_input
        gen_input = torch.cat((noise, image.view(image.shape[0], -1)), -1)
        gen_ctx_prompt = self.model(gen_input)

        gen_ctx_prompt = gen_ctx_prompt.reshape(image.shape[0], n_ctx, 512)

        return gen_ctx_prompt


class Discriminator(nn.Module):
    def __init__(self, cfg, n_ctx):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_ctx * 512 + cfg.FEAT_DIM, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

        # Initialization with He
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, ctx_prompt, image):
        # concatenate context prompt and image label to produce disc_input
        disc_input = torch.cat((ctx_prompt.view(ctx_prompt.shape[0], -1), image.view(image.shape[0], -1)), -1)
        validity = self.model(disc_input)
        return validity


# stage 2nd: train the CPG model
@TRAINER_REGISTRY.register()
class SPG_CGAN(TrainerX):
    """Soft Prompt Generation with CGAN (SPG_CGAN).
    """    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # loading CLIP model
        self.clip_model = load_clip_to_cpu(cfg).to(self.device)

        self.n_ctx = cfg.TRAINER.SPG.N_CTX
        self.ctx_init = cfg.TRAINER.SPG.CTX_INIT
        if self.ctx_init:
            self.ctx_init = self.ctx_init.replace("_", " ")
            self.n_ctx = len(self.ctx_init.split(" "))
        
        print("Building SPG_CGAN")
        self.gmodel = Generator(cfg, self.n_ctx)
        self.dmodel = Discriminator(cfg, self.n_ctx)
        
        self.gmodel.to(self.device)
        self.dmodel.to(self.device)
        
        self.optimizer_G = torch.optim.AdamW(self.gmodel.parameters(), lr=cfg.OPTIM.LR, weight_decay=1e-4)
        self.optimizer_D = torch.optim.AdamW(self.dmodel.parameters(), lr=cfg.OPTIM.LR, weight_decay=1e-4)
        self.sched_G = build_lr_scheduler(self.optimizer_G, cfg.OPTIM)
        self.sched_D = build_lr_scheduler(self.optimizer_D, cfg.OPTIM)
        self.register_model("generator", self.gmodel, self.optimizer_G, self.sched_G)
        self.register_model("discriminator", self.dmodel, self.optimizer_D, self.sched_D)

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            
        #     self.gmodel = nn.DataParallel(self.gmodel)
        #     self.dmodel = nn.DataParallel(self.dmodel)

        self.best_prompts = {}

        for i in range(len(cfg.ALL_DOMAINS)):
            prompt_dir = prompt_dir = 'prompt_labels' + '/' + self.cfg.DATASET.NAME.split('_')[1] + '/' + self.cfg.MODEL.BACKBONE.NAME.replace('/', '') + '/' + 'seed_' + str(1)
            prompts_path = os.path.join(prompt_dir, self.cfg.DATASET.NAME.split('_')[1] + '_CoOp_' + self.cfg.ALL_DOMAINS[i] + '.pt')
            prompt_label = torch.load(prompts_path).to(self.device)
            self.best_prompts[i] = prompt_label

    def forward_backward(self, batch):
        cfg = self.cfg
        image, label, domain = self.parse_batch_train(batch)

        # select MSELoss as the loss function
        adversarial_loss = torch.nn.MSELoss()
        adversarial_loss.to(self.device)

        # image_features = self.clip_model.visual(image.type(self.clip_model.dtype))
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.to(torch.float32)
        image_features = image_features.detach()

        real = torch.ones((image_features.size(0), 1), dtype=torch.float32, requires_grad=False).to(self.device)
        fake = torch.zeros((image_features.size(0), 1), dtype=torch.float32, requires_grad=False).to(self.device)

        data_loader = self.train_loader_x
        randnum = self.fake_list[self.batch_idx]

        for batch_idx_fake, batch_fake in enumerate(data_loader):
            if(batch_idx_fake == randnum):
                fake_image, fake_label, fake_domain = self.parse_batch_train(batch_fake)
                break

        self.gmodel.train()
        self.dmodel.train()

        """Train Discriminator
        """
        self.optimizer_D.zero_grad()

        single_batch_prompts = torch.unsqueeze(self.best_prompts[0], 0)
        batch_prompts = torch.clone(single_batch_prompts)
        for imgnum in range(image.shape[0]):
            if(imgnum < image.shape[0]-1):
                batch_prompts = torch.cat((batch_prompts, single_batch_prompts), dim=0)
            batch_prompts[imgnum] = self.best_prompts[int(domain[imgnum])]

        validity_real = self.dmodel(batch_prompts, image_features)

        d_real_loss = adversarial_loss(validity_real, real)
        d_real_loss.backward()

        # fake_image_features = self.clip_model.visual(fake_image.type(self.clip_model.dtype))
        fake_image_features = self.clip_model.encode_image(fake_image)
        fake_image_features = fake_image_features / fake_image_features.norm(dim=-1, keepdim=True)
        fake_image_features = fake_image_features.detach()

        
        gen_prompt = self.gmodel(fake_image_features, self.n_ctx)
        
        validity_fake = self.dmodel(gen_prompt.detach(), fake_image_features)

        d_fake_loss = adversarial_loss(validity_fake, fake)
        d_fake_loss.backward()


        # d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss = d_real_loss + d_fake_loss

        if cfg.GRAD_CLIP:
            max_norm_weight = cfg.D_MAX_NORM_WEIGHT
            max_norm_bias = cfg.D_MAX_NORM_BIAS
            max_norm_last = cfg.D_MAX_NORM_LAST

            d_gradients = [(p[1].grad, p[0]) for p in self.dmodel.named_parameters()]
            for grad, name in d_gradients:
                norm = torch.norm(grad)
                if "0.weight" in name or "0.bias" in name or "8.weight" in name or "8.bias" in name:
                    if norm > max_norm_last:
                        parameters = [p[1] for p in self.dmodel.named_parameters() if p[0] == name]
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm_last)
                elif "weight" in name:
                    if norm > max_norm_weight:
                        parameters = [p[1] for p in self.dmodel.named_parameters() if p[0] == name]
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm_weight)
                else:
                    if norm > max_norm_bias:
                        parameters = [p[1] for p in self.dmodel.named_parameters() if p[0] == name]
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm_bias)


        self.optimizer_D.step()

        
        """Train Generator
        """
        self.optimizer_G.zero_grad()
        validity_fake = self.dmodel(gen_prompt, fake_image_features)

        g_loss = adversarial_loss(validity_fake, real)

        g_loss.backward()

        if cfg.GRAD_CLIP:
            max_norm_weight = cfg.G_MAX_NORM_WEIGHT
            max_norm_bias = cfg.G_MAX_NORM_BIAS
            max_norm_bias_last = cfg.G_MAX_NORM_BIAS_LAST

            g_gradients = [(p[1].grad, p[0]) for p in self.gmodel.named_parameters()]
            for grad, name in g_gradients:
                norm = torch.norm(grad)
                if "weight" in name:
                    if norm > max_norm_weight:
                        parameters = [p[1] for p in self.gmodel.named_parameters() if p[0] == name]
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm_weight)
                elif "9.bias" in name or "11.bias" in name:
                    if norm > max_norm_bias_last:
                        parameters = [p[1] for p in self.gmodel.named_parameters() if p[0] == name]
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm_bias_last)
                else:
                    if norm > max_norm_bias:
                        parameters = [p[1] for p in self.gmodel.named_parameters() if p[0] == name]
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm_bias)

        self.optimizer_G.step()


        loss_summary = {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.gmodel.eval()
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"\nEvaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)
            
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_encoder = BaseTextEncoder(self.clip_model)

            n_ctx = self.cfg.TRAINER.SPG.N_CTX
            ctx_init = self.cfg.TRAINER.SPG.CTX_INIT
            dtype = self.clip_model.dtype
            if ctx_init:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                prompt_prefix = ctx_init

            else:
                # initializing a generic context
                prompt_prefix = " ".join(["X"] * n_ctx)

            classnames = self.dm.dataset.classnames
            n_cls = len(classnames)

            classnames = [name.replace("_", " ") for name in classnames]
            cls_prompts = [prompt_prefix + " " + name + "." for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in cls_prompts])

            goutput = self.gmodel(image_features, n_ctx)
            batch_prompts = torch.ones(n_cls, goutput.size(1), goutput.size(2)).to(self.device)
            for idnum in range(n_cls):
                batch_prompts[idnum] = torch.clone(goutput[0])
            gen_prompt = torch.tensor(batch_prompts, dtype=torch.float16)

            token_embedding = self.clip_model.token_embedding
            
            tokenized_prompts = tokenized_prompts.to(self.device)
            with torch.no_grad():
                embedding = token_embedding(tokenized_prompts).type(self.clip_model.dtype)
            self.token_prefix = embedding[:, :1, :]  # SOS
            self.token_suffix = embedding[:, 1 + n_ctx :, :]  # CLS, EOS

            self.class_token_position = self.cfg.TRAINER.SPG.CLASS_TOKEN_POSITION

            if gen_prompt.dim() == 2:
                gen_prompt = gen_prompt.unsqueeze(0).expand(n_cls, -1, -1)

            prefix = self.token_prefix.to(self.device)
            suffix = self.token_suffix.to(self.device)

            gen_prompts = self.construct_prompts(gen_prompt, prefix, suffix)

            tokenized_prompts = tokenized_prompts.to(self.device)

            text_features = text_encoder(gen_prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            output = image_features.float() @ text_features.float().t()
            
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def construct_prompts(self, ctx, prefix, suffix):
        '''
        dim0 is either batch_size (during training) or n_cls (during testing)
        ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        '''
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        
        return prompts

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)
        return input, label, domain

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            # if name == "prompt_learner":
            model_path = os.path.join(directory, name, model_file)

            if not os.path.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            optimizer = checkpoint["optimizer"]
            scheduler = checkpoint["scheduler"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            # self._models[name].load_state_dict(state_dict, strict=False)
            self._models[name].load_state_dict(state_dict)
            self._optims[name].load_state_dict(optimizer)
            self._scheds[name].load_state_dict(scheduler)

    def load_checkpoint(self, fpath):
        if fpath is None:
            raise ValueError("File path is None")

        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))

        map_location = self.device if torch.cuda.is_available() else "cpu"

        try:
            checkpoint = torch.load(fpath, map_location=map_location)

        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )

        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise

        return checkpoint

    def before_train(self):
        
        num_batches = len(self.train_loader_x)
        self.fake_list = [i for i in range(num_batches)]

        # Initialize summary writer
        writer_dir = os.path.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def before_epoch(self):
        random.shuffle(self.fake_list)

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % (self.cfg.TRAIN.PRINT_FREQ) == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        curr_result = self.test('val')
        is_best = curr_result > self.best_result
        if is_best:
            self.best_result = curr_result
            self.best_epoch = self.epoch

            self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
            best_val_dir = os.path.join(self.output_dir, 'best_val.pt')
            torch.save(self.gmodel, best_val_dir)

        print('******* Best val acc: {:.1f}%, epoch: {} *******'.format(self.best_result, self.best_epoch+1))

        n_iter = self.epoch
        self.write_scalar("train/val_acc", curr_result, n_iter)
        
        self.set_model_mode("train")
        if self.cfg.SAVE_MODEL and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)
    
    
    def after_train(self):
        print("----------Finish training----------")
        print("Deploy the best model")
        model_dir = os.path.join(self.output_dir, 'best_val.pt')
        self.gmodel = torch.load(model_dir).to(self.device)
        curr_test_result = self.test('test')
        print('******* Test acc: {:.1f}% *******'.format(curr_test_result))


        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()
