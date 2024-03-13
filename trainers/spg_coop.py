import os
import numpy as np
from torch.cuda.amp import GradScaler, autocast

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import dassl
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.utils.tools import mkdir_if_missing
from dassl.optim import build_optimizer, build_lr_scheduler

from .basedg import *
from utils.clip_part import *


_tokenizer = _Tokenizer()


class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.SPG.N_CTX
        ctx_init = cfg.TRAINER.SPG.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        if ctx_init: # use given words to initialize context vectors 
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:   # random initialization
            if cfg.TRAINER.SPG.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.SPG.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts
    
    
class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompts = torch.tensor(0)

    def forward(self, image):
        prompts = self.prompt_learner()
        self.prompts = prompts
        
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    
    
@TRAINER_REGISTRY.register()
class SPG_CoOp(BaseDG):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        if not cfg.TEST.NO_TEST:
            self.test_best_result = -np.inf

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})...")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.SPG.PREC == "fp32" or cfg.TRAINER.SPG.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.SPG.PREC == "amp" else None
    
    def forward_backward(self, batch):
        images, labels = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.SPG.PREC
        if prec == "amp":
            with autocast():
                output = self.model(images)
                loss = F.cross_entropy(output, labels)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(images)
            loss = F.cross_entropy(output, labels)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, labels)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        curr_result = self.test(split="val")
        is_best = curr_result > self.best_result
        if is_best:
            self.best_result = curr_result
            self.best_epoch = self.epoch
            if self.cfg.SAVE_MODEL:
                self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
                
                prompts_path = os.path.join(self.output_dir, 'prompt_learner', f'{self.cfg.DATASET.NAME}_{self.cfg.TARGET_DOMAIN}.pt')
                torch.save(self.model.prompt_learner.ctx, prompts_path)
        
                self.n_ctx = self.cfg.TRAINER.SPG.N_CTX
                prompt_dir = 'prompt_labels' + '/' + self.cfg.DATASET.NAME.split('_')[0] + '/' + self.cfg.MODEL.BACKBONE.NAME.replace('/', '') + '/' + 'seed_' + str(self.cfg.SEED)
                mkdir_if_missing(prompt_dir)
                prompts_path = os.path.join(prompt_dir, f'{self.cfg.DATASET.NAME}_{self.cfg.TARGET_DOMAIN}.pt')
                prompts = self.model.prompts[0]
                prompt = prompts[1 : self.model.prompt_learner.n_ctx+1, :]
                torch.save(prompt, prompts_path)

                print(f'Prompt saved to {prompts_path}')
        print('Domain {} val best acc: {:.1f}%, best epoch: {}'.format(self.cfg.TARGET_DOMAIN, self.best_result, self.best_epoch+1))
        
        self.set_model_mode("train")
        if self.cfg.SAVE_MODEL and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)

