import json
import os.path as osp
import dassl
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


def read_split(root_path, split_path):
    def _convert(items):
        out = []
        for impath, label, domain, classname in items:
            impath = osp.join(root_path, impath)
            item = Datum(impath=impath, label=int(label), domain=domain, classname=classname)
            out.append(item)
            
        return out
    
    with open(split_path, "r") as f:
        split = json.load(f)
    train = _convert(split["train"])
    val = _convert(split["val"])
    
    return train, val


@DATASET_REGISTRY.register()
class SINGLE_OfficeHome(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home_dg"
    domains = ["art", "clipart", "product", "real_world"]
    data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"

    def __init__(self, cfg):
        self.root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "spg_coop_splits")

        if not osp.exists(self.dataset_dir):
            dst = osp.join(self.root, "office_home_dg.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train, val, test = self._read_data(cfg.SOURCE_DOMAIN, cfg.TARGET_DOMAINS)

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, source_domain, target_domains):
        train, val, test = [], [], []

        for domain, dname in enumerate(target_domains):
            split_path = osp.join(self.split_dir, dname + "_train_val_split.json")
            split_train, split_val = read_split(self.root, split_path)
            
            test.extend(split_train)
            test.extend(split_val)

        split_path = osp.join(self.split_dir, source_domain + "_train_val_split.json")
        split_train, split_val = read_split(self.root, split_path)

        train.extend(split_train)
        val.extend(split_val)

        return train, val, test
