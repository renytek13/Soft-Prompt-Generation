import json
import os.path as osp

import dassl
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


def read_split(root_path, split_path, error_paths):
    def _convert(items):
        out = []
        for impath, label, domain, classname in items:
            impath = osp.join(root_path, impath)
            if impath == error_paths:
                continue
            item = Datum(impath=impath, label=int(label), domain=domain, classname=classname)
            out.append(item)
            
        return out
    
    with open(split_path, "r") as f:
        split = json.load(f)
    train = _convert(split["train"])
    val = _convert(split["val"])
    
    return train, val


@DATASET_REGISTRY.register()
class PACS_ABLATION(DatasetBase):
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
        self.root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        # self.split_dir = osp.join(self.dataset_dir, "spg_coop_splits", str(cfg.SEED))
        self.split_dir = osp.join(self.dataset_dir, "spg_coop_splits", "1")

        if not osp.exists(self.dataset_dir):
            dst = osp.join(self.root, "pacs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train, val, test = self._read_data(cfg.SOURCE_DOMAINS, cfg.TARGET_DOMAIN, cfg.TRAIN_SAMPLE)

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, source_domains, target_domain, train_sample):
        train, val, test = [], [], []
        error_paths = osp.join(self.image_dir, self._error_paths)

        for domain, dname in enumerate(source_domains):
            split_path = osp.join(self.split_dir, dname + "_train_val_split.json")
            split_train, split_val = read_split(self.root, split_path, error_paths)

            new_train = int(len(split_train) * train_sample)
            
            train.extend([split_train[i] for i in range(new_train)])
            val.extend(split_val)

        split_path = osp.join(self.split_dir, target_domain + "_train_val_split.json")
        split_train, split_val = read_split(self.root, split_path, error_paths)

        test.extend(split_train)
        test.extend(split_val)

        return train, val, test
