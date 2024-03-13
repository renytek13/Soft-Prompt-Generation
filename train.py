import argparse
import torch

import dassl
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
from datasets import *
from trainers import *


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    
    cfg.MODEL_DIR = ""
    if args.model_dir:
        cfg.MODEL_DIR = args.model_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    
    # embedding dimension size for image feature
    if cfg.MODEL.BACKBONE.NAME == "RN50":
        cfg.FEAT_DIM = 1024
    elif cfg.MODEL.BACKBONE.NAME == "ViT-B/16":
        cfg.FEAT_DIM = 512

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.n_classes:
        cfg.DATASET.NUM_CLASSES = args.n_classes

    if args.latent_dim:
        cfg.DATASET.LATENT_DIM = args.latent_dim
    
    if args.save:
        cfg.SAVE_MODEL = args.save
    
    if args.gpu:
        cfg.GPU = args.gpu

    if args.lr:
        cfg.OPTIM.LR = args.lr

    if args.weight_decay:
        cfg.OPTIM.WEIGHT_DECAY = args.weight_decay
    
    if args.warmup_epoch:
        cfg.OPTIM.WARMUP_EPOCH = args.warmup_epoch

    if args.g_max_norm_weight:
        cfg.G_MAX_NORM_WEIGHT = args.g_max_norm_weight

    if args.g_max_norm_bias:
        cfg.G_MAX_NORM_BIAS = args.g_max_norm_bias

    if args.g_max_norm_bias_last:
        cfg.G_MAX_NORM_BIAS_LAST = args.g_max_norm_bias_last

    if args.d_max_norm_weight:
        cfg.D_MAX_NORM_WEIGHT = args.d_max_norm_weight

    if args.d_max_norm_bias:
        cfg.D_MAX_NORM_BIAS = args.d_max_norm_bias

    if args.d_max_norm_last:
        cfg.D_MAX_NORM_LAST = args.d_max_norm_last
        
    if args.target_domains:
        if "PACS" in cfg.DATASET.NAME:
            DOMAINS = {'a': "art_painting", 'c':"cartoon", 'p':"photo", 's':"sketch"}
        elif "VLCS" in cfg.DATASET.NAME:
            DOMAINS = {'c': "caltech", 'l':"labelme", 'p':"pascal", 's':"sun"}
        elif "OfficeHome" in cfg.DATASET.NAME:
            DOMAINS = {'a': "art", 'c':"clipart", 'p':"product", 'r':"real_world"}
        elif "TerraIncognita" in cfg.DATASET.NAME:
            DOMAINS = {'1':"location_100", '2': "location_38", '3':"location_43", '4':"location_46"}
        elif "DomainNet" in cfg.DATASET.NAME:
            DOMAINS = {'c': "clipart", 'i':"infograph", 'p':"painting", 'q':"quickdraw", 'r':"real", 's':"sketch"}
        else:
            raise ValueError
        
        cfg.ALL_DOMAINS = list(DOMAINS.keys())
        
        cfg.TARGET_DOMAIN = args.target_domains[0]
        cfg.DATASET.TARGET_DOMAINS = [DOMAINS[cfg.TARGET_DOMAIN]]
        
        DOMAINS.pop(cfg.TARGET_DOMAIN)
        cfg.SOURCE_DOMAINS = list(DOMAINS.keys())
        cfg.DATASET.SOURCE_DOMAINS = list(DOMAINS.values())
    
    if args.source_domains:
        if "PACS" in cfg.DATASET.NAME:
            DOMAINS = {'a': "art_painting", 'c':"cartoon", 'p':"photo", 's':"sketch"}
        elif "VLCS" in cfg.DATASET.NAME:
            DOMAINS = {'c': "caltech", 'l':"labelme", 'p':"pascal", 's':"sun"}
        elif "OfficeHome" in cfg.DATASET.NAME:
            DOMAINS = {'a': "art", 'c':"clipart", 'p':"product", 'r':"real_world"}
        elif "TerraIncognita" in cfg.DATASET.NAME:
            DOMAINS = {'1':"location_100", '2': "location_38", '3':"location_43", '4':"location_46"}
        elif "DomainNet" in cfg.DATASET.NAME:
            DOMAINS = {'c': "clipart", 'i':"infograph", 'p':"painting", 'q':"quickdraw", 'r':"real", 's':"sketch"}
        else:
            raise ValueError
        
        cfg.ALL_DOMAINS = list(DOMAINS.keys())
        
        cfg.SOURCE_DOMAIN = args.source_domains[0]
        cfg.DATASET.SOURCE_DOMAINS = [DOMAINS[cfg.SOURCE_DOMAIN]]
        
        DOMAINS.pop(cfg.SOURCE_DOMAIN)
        cfg.TARGET_DOMAINS = list(DOMAINS.keys())
        cfg.DATASET.TARGET_DOMAINS = list(DOMAINS.values())
        

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    
    cfg.MODEL.BACKBONE.PATH = "./assets"        # path of pretrained CLIP model
    cfg.TEST.FINAL_MODEL = "best_val"
    cfg.LATENT_DIM = 100    # size of latent space
    cfg.GRAD_CLIP = False
    cfg.D_MAX_NORM_WEIGHT = 5e-2
    cfg.D_MAX_NORM_BIAS = 5e-1
    cfg.D_MAX_NORM_LAST = 5
    cfg.G_MAX_NORM_WEIGHT = 5e-3
    cfg.G_MAX_NORM_BIAS = 5e-8
    cfg.G_MAX_NORM_BIAS_LAST = 1
    
    if 'CLIP' in args.trainer:
        cfg.TRAINER.CLIP = CN()
        cfg.TRAINER.CLIP.PREC = "fp16"  # fp16, fp32, amp 
        
    elif 'SPG' in args.trainer:
        cfg.TRAINER.SPG = CN()
        cfg.TRAINER.SPG.PREC = "fp16"       # fp16, fp32, amp  
        cfg.TRAINER.SPG.N_CTX = 16          # number of text context vectors
        cfg.TRAINER.SPG.CSC = False         # class-specific context
        cfg.TRAINER.SPG.CTX_INIT = ""       # initialization words
        cfg.TRAINER.SPG.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
        if 'GAN' in args.trainer:
            cfg.TRAINER.SPG.PREC = "fp32"       # fp16, fp32, amp  
            cfg.OPTIM.NAME = 'AdamW'
            cfg.OPTIM.WEIGHT_DECAY = 1e-4
        

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=None)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/opt/data/private/OOD_data", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="output directory")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to config file for dataset setup")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    
    parser.add_argument("--trainer", type=str, default="SPG_CGAN", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")

    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--warmup_epoch", type=int, default=4, help="warmup epoch")
    parser.add_argument("--save", type=str, default=True, help="need to save model")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    
    parser.add_argument("--g_max_norm_weight", type=float, default=5e-3, help="g_max_norm_weight")
    parser.add_argument("--g_max_norm_bias", type=float, default=5e-8, help="g_max_norm_bias")
    parser.add_argument("--g_max_norm_bias_last", type=float, default=1, help="g_max_norm_bias_last")
    parser.add_argument("--d_max_norm_weight", type=float, default=5e-2, help="d_max_norm_weight")
    parser.add_argument("--d_max_norm_bias", type=float, default=5e-1, help="d_max_norm_bias")
    parser.add_argument("--d_max_norm_last", type=float, default=5, help="d_max_norm_last")

    args = parser.parse_args()
    
    main(args)
