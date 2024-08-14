import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers_baseline.basedg import *
from utils.clip_part import *
from utils.templates import CUSTOM_TEMPLATES
from utils.visual_prompt import *

_tokenizer = _Tokenizer()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.VP.N_CTX
        ctx_init = cfg.TRAINER.VP.CTX_INIT
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_prefix = ctx_init
        else:
            # random initialization
            prompt_prefix = " ".join(["X"] * n_ctx)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.text_encoder = Simple_TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        if cfg.TRAINER.VP.TYPE == "random":
            random_prompter = RandomPatchPrompter(cfg)
            self.prompter = random_prompter
        elif cfg.TRAINER.VP.TYPE == "fix":
            fix_prompter = FixedPatchPrompter(cfg)
            self.prompter = fix_prompter
        elif cfg.TRAINER.VP.TYPE == "pad":
            pad_prompter = PadPrompter(cfg)
            self.prompter = pad_prompter
        else:
            raise ValueError('VP type is wrong!')

    def forward(self, image):
        # self.text_features = self.text_encoder(self.tokenized_prompts)
        self.text_features = self.text_encoder(self.tokenized_prompts.to(self.logit_scale.device))
        
        prompter_image = self.prompter(image)
        self.image_features = self.image_encoder(prompter_image.type(self.dtype))
        self.image_features = self.image_features / self.image_features.norm(dim=-1, keepdim=True)
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * self.image_features @ self.text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class VP(BaseDG):
    '''Visual Prompting (VP)
    
    '''
    def load_clip(self, cfg):
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        model = model.to(self.device)

        return model
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        if torch.cuda.is_available() and cfg.USE_CUDA:
            if len(cfg.GPU) == 1:
                self.device = torch.device("cuda:{}".format(cfg.GPU))
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.best_test_result = -np.inf
        self.best_val_test_result = -np.inf

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        # clip_model = load_clip_to_cpu(cfg)
        clip_model = self.load_clip(cfg)

        if cfg.TRAINER.VP.PREC == "fp32" or cfg.TRAINER.VP.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            if "prompter" not in name:
                param.requires_grad_(False)
                
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")
        print("# params: {:,}".format(count_num_param(self.model.prompter)))

        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        

        self.model.to(self.device)
        
        # NOTE: only give prompter to the optimizer
        self.optim = build_optimizer(self.model.prompter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompter", self.model.prompter, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.VP.PREC == "amp" else None

    def forward_backward(self, batch):
        images, labels = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.VP.PREC
        
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
