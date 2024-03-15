import os
import json
import random

import dassl
from dassl.utils import mkdir_if_missing, listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


def split_data(cfg, image_dir):
    domains = listdir_nohidden(image_dir)
    domains.sort()
    print(f'VLCS domains: {domains}, use {cfg.DATASET.TARGET_DOMAINS[0].upper()} to obtain best prompt as label.')
    
    domain_dir = os.path.join(image_dir, cfg.DATASET.TARGET_DOMAINS[0].upper(), 'full')
    classnames = ['bird', 'car', 'chair', 'dog', 'person']
    classnames_index = listdir_nohidden(domain_dir)
    classnames_index.sort()
    print(('VLCS has {} categories: {}').format(len(classnames), classnames))
    
    # p_trn = 0.85
    p_trn = 0.7
    print(f"Splitting into {p_trn:.0%} train and {1 - p_trn:.0%} val")

    train, val = [], []
    for label, class_name in enumerate(classnames):
        class_path = os.path.join(domain_dir, classnames_index[label])
        imnames = listdir_nohidden(class_path)
        n_total = len(imnames)
        n_train = round(n_total * p_trn)
        random.shuffle(imnames)
        
        items = []
        for imname in imnames:
            impath = os.path.join(class_path, imname)
            item = Datum(
                impath=impath,
                label=label,
                domain=domains.index(cfg.DATASET.TARGET_DOMAINS[0].upper()),
                classname=class_name
            )
            items.append(item)
        train.extend(items)
        # train.extend(items[:n_train])
        # val.extend(items[n_train:])
    
    domain_dir = os.path.join(image_dir, cfg.DATASET.TARGET_DOMAINS[0].upper(), 'test')
    classnames = ['bird', 'car', 'chair', 'dog', 'person']
    classnames_index = listdir_nohidden(domain_dir)
    classnames_index.sort()
    
    for label, class_name in enumerate(classnames):
        class_path = os.path.join(domain_dir, classnames_index[label])
        imnames = listdir_nohidden(class_path)
        random.shuffle(imnames)
        
        items = []
        for imname in imnames:
            impath = os.path.join(class_path, imname)
            item = Datum(
                impath=impath,
                label=label,
                domain=domains.index(cfg.DATASET.TARGET_DOMAINS[0].upper()),
                classname=class_name
            )
            items.append(item)
        val.extend(items)
        
    return train, val, val


def read_split(root_path, split_path):
        def _convert(items):
            out = []
            for impath, label, domain, classname in items:
                impath = os.path.join(root_path, impath)
                item = Datum(impath=impath, label=int(label), domain=domain, classname=classname)
                out.append(item)
                
            return out
        
        with open(split_path, "r") as f:
            split = json.load(f)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        
        return train, val, test


def save_split(train, val, test, root_path, split_path):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                domain = item.domain
                label = item.label
                classname = item.classname
                impath = impath.replace(root_path, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append([impath, label, domain, classname])
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)
        split = {"train": train, "val": val, "test": test}

        with open(split_path, "w") as f:
            json.dump(split, f, indent=4, separators=(",", ": "))
        print(f"Saved split to {split_path}")


@DATASET_REGISTRY.register()
class VLCS_CoOp(DatasetBase):

    dataset_dir = "VLCS"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        
        self.split_dir = os.path.join(self.dataset_dir, "spg_coop_splits")
        mkdir_if_missing(self.split_dir)
        self.split_path = os.path.join(self.split_dir, f"{cfg.TARGET_DOMAIN}_train_val_split.json")
        
        if os.path.exists(self.split_path):
            train, val, test = read_split(root, self.split_path)
        else:
            train, val, test = split_data(cfg, self.image_dir)
            save_split(train, val, test, root, self.split_path)

        super().__init__(train_x=train, val=val, test=test)
        
        