import os
import json
import random

import dassl
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, listdir_nohidden


def split_data(cfg, image_dir):
    domains = listdir_nohidden(image_dir)
    domains.sort()
    print(f'PACS domains: {domains}, use {cfg.DATASET.TARGET_DOMAINS} to obtain best prompt as label.')
    
    domain_dir = os.path.join(image_dir, cfg.DATASET.TARGET_DOMAINS[0])
    class_names = listdir_nohidden(domain_dir)
    class_names.sort()
    print(('PACS has {} categories: {}').format(len(class_names), class_names))
    
    # p_trn = 0.85
    p_trn = 0.7
    print(f"Splitting into {p_trn:.0%} train and {1 - p_trn:.0%} val")

    train, val = [], []
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(domain_dir, class_name)
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
                domain=domains.index(cfg.DATASET.TARGET_DOMAINS[0]),
                classname=class_name
            )
            items.append(item)
        train.extend(items[:n_train])
        val.extend(items[n_train:])
        
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
class PACS_CoOp(DatasetBase):
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
    """

    dataset_dir = "pacs"
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    data_url = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"
    # the following images contain errors and should be ignored
    _error_paths = "sketch/dog/n02103406_4068-1.png"

    def __init__(self, cfg):
        self.root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        if not os.path.exists(self.dataset_dir):
            dst = os.path.join(self.root, "pacs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        
        self.split_dir = os.path.join(self.dataset_dir, "spg_coop_splits", str(cfg.SEED))
        mkdir_if_missing(self.split_dir)
        self.split_path = os.path.join(self.split_dir, f"{cfg.TARGET_DOMAIN}_train_val_split.json")
        
        if os.path.exists(self.split_path):
            train, val, test = read_split(self.root, self.split_path)
        else:
            train, val, test = split_data(cfg, self.image_dir)
            save_split(train, val, test, self.root, self.split_path)

        super().__init__(train_x=train, val=val, test=test)
        
        