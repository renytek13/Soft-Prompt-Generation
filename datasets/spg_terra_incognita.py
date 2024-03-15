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
class SPG_TerraIncognita(DatasetBase):
    """Terra-Incognita.

    Statistics:
        - Around 24,330 images.
        - 10 classes related to terra objects.
        - 4 domains: location_38, location_43, location_46, location_100.
        - URL: http://lila.science/datasets/caltech-camera-traps.

    Reference:
        - Sara Beery, Grant Van Horn, Pietro Perona. Recognition in Terra Incognita. 
        Proceedings of the 15th European Conference on Computer Vision (ECCV 2018).
    """

    dataset_dir = "terra"
    domains = ["location_38", "location_43", "location_46", "location_100"]

    def __init__(self, cfg):
        self.root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "spg_coop_splits")

        # if not osp.exists(self.dataset_dir):
        #     dst = osp.join(self.root, "office_home_dg.zip")
        #     self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train, val, test = self._read_data(cfg.SOURCE_DOMAINS, cfg.TARGET_DOMAIN)

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, source_domains, target_domain):
        train, val, test = [], [], []

        for domain, dname in enumerate(source_domains):
            split_path = osp.join(self.split_dir, dname + "_train_val_split.json")
            split_train, split_val = read_split(self.root, split_path)
            
            train.extend(split_train)
            val.extend(split_val)

        split_path = osp.join(self.split_dir, target_domain + "_train_val_split.json")
        split_train, split_val = read_split(self.root, split_path)

        test.extend(split_train)
        test.extend(split_val)

        return train, val, test
