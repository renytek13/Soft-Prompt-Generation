import json
import os.path as osp
import dassl
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, listdir_nohidden


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
class CROSS(DatasetBase):

    def __init__(self, cfg):
        self.root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))

        # if not osp.exists(self.dataset_dir):
        #     dst = osp.join(self.root, "office_home_dg.zip")
        #     self.download_data(self.data_url, dst, from_gdrive=True)

        train, val, test = self._read_data(cfg.DATASET.SOURCE_DATASETS)

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, source_dataset):
        train, val = [], []

        dataset_dir = source_dataset[0]
        dataset_dir = osp.join(self.root, dataset_dir)
        image_dir = osp.join(dataset_dir, "images")
        split_dir = osp.join(dataset_dir, "spg_coop_splits")

        source_domains = listdir_nohidden(split_dir)
        source_domains.sort()
        for domain, dfile in enumerate(source_domains):
            split_path = osp.join(split_dir, dfile)
            split_train, split_val = read_split(self.root, split_path)
            
            train.extend(split_train)
            val.extend(split_val)

        return train, val, val
