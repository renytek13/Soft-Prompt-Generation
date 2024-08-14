## How to install datasets

We recommend that all datasets be placed under the same folder `$DATA` for ease of management, with the file structure organized as follows:
```
$DATA/
|–– domainnet/
|–– office_home_dg/
|–– PACS/
|–– terra/
|–– VLCS/
```

The dataset list is as follows:
- [DomainNet](#domainnet)
- [OfficeHome](#officehome)
- [PACS](#pacs)
- [TerraIncognita](#terraincognita)
- [VLCS](#vlcs)

To ensure reproducibility and fairness in future work, we provide a fixed set of training set/validation set/test set splits for all datasets, each of which is prepared as described below.

### DomainNet
- Create a folder named `domainnet/` under `$DATA`.
- Create `images/` under `domainnet/`.
- Download clipart domain images from http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip.
- Download infograph domain images from http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip.
- Download painting domain images from http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip.
- Download quickdraw domain images from http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip.
- Download real domain images from http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip.
- Download sketch domain images from http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip.
- Extract the above downloaded images to `$DATA/DomainNet/images`.
- Download `spg_coop_splits.zip` from this [link](**...(NEED FINISHED)**) and extract the folder under `$DATA/DomainNet`.

The specific directory structure is as follows.
```
domainnet/
|–– images/
|   |–– clipart/
|   |–– infograph/
|   |–– painting/
|   |–– quickdraw/
|   |–– real/
|   |–– sketch/
|–– spg_coop_splits/
```

### OfficeHome
- Create a folder named `office_home_dg/` under `$DATA`.
- Download `office_home_dg.zip` from https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa and extract the folder `office_home_dg/`. Then rename the folder `office_home_dg` to `images` and put it under `OfficeHome/`.
- Download `spg_coop_splits.zip` from this [link](**...(NEED FINISHED)**) and and extract the folder under `$DATA/OfficeHome`.

The specific directory structure is as follows
```
office_home_dg/
|–– images/
|   |–– art/
|   |–– clipart/
|   |–– product/
|   |–– real_world/
|–– spg_coop_splits/
```

### PACS
- Create a folder named `PACS/` under `$DATA`.
- Download `pacs.zip` from https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE and extract the folder `pacs/images/`. Then put the folder `images/` under `PACS/`.
- Download `spg_coop_splits.zip` from this [link](**...(NEED FINISHED)**) and and extract the folder under `$DATA/PACS`.

The specific directory structure is as follows
```
PACS/
|–– images/
|   |–– art_painting/
|   |–– cartoon/
|   |–– photo/
|   |–– sketch/
|–– spg_coop_splits/
```

### TerraIncognita
- Create a folder named `terra/` under `$DATA`.
- Download **...zip(NEED FINISHED)** from **...(NEED FINISHED)** and extract the folder. Then put the folder `images/` under `TerraIncognita/`.
- Download `spg_coop_splits.zip` from this [link](**...(NEED FINISHED)**) and and extract the folder under `$DATA/TerraIncognita`.

The specific directory structure is as follows
```
terra/
|–– images/
|   |–– location_38/
|   |–– location_43/
|   |–– location_46/
|   |–– location_100/
|–– spg_coop_splits/
```

### VLCS
- Create a folder named `VLCS/` under `$DATA`.
- Download `vlcs.zip` from https://drive.google.com/uc?id=1r0WL5DDqKfSPp9E3tRENwHaXNs1olLZd and extract the folder `VLCS/`. Then rename the folder `VLCS` to `images` and put it under `VLCS/`.
- Download `spg_coop_splits.zip` from this [link](**...(NEED FINISHED)**) and and extract the folder under `$DATA/VLCS`.

The specific directory structure is as follows
```
VLCS/
|–– images/
|   |–– CALTECH/
|   |–– LABELME/
|   |–– PASCAL/
|   |–– SUN/
|–– spg_coop_splits/
```
