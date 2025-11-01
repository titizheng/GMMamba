# GMMamba: Group Masking Mamba for Whole Slide Image Classification


[Tingting Zheng](https://scholar.google.com/citations?user=AJ5zl-wAAAAJ&hl=zh-CN), [Hongxun Yao](https://scholar.google.com/citations?user=aOMFNFsAAAAJ),, 
[Kui jiang](https://scholar.google.com/citations?user=AbOLE9QAAAAJ&hl=en&oi=ao),  [Sicheng Zhao](https://scholar.google.com/citations?user=LJiQRJIAAAAJ&hl=zh-CN&oi=ao). [Yi Xiao](https://scholar.google.com/citations?user=e3a4aG0AAAAJ).

 


**Abstract:** Recent advances in selective state space model (Mamba) have shown great promise in whole slide image (WSI) classification. 
Despite this, WSIs contain explicit local redundancy (similar patches) and irrelevant regions (uninformative instances), posing significant challenges for Mamba-based multi-instance learning (MIL) methods in capturing global representations. 
Furthermore, bag-level approaches struggle to extract critical features from all instances, while group-level methods fail to adequately account for tumor dispersion and intrinsic correlations across groups, leading to suboptimal global representations. 
To address these issues, we propose group masking Mamba (GMMamba), a novel framework that combines two elaborate modules: (1) intra-group masking Mamba (IMM) for selective instance exploration within groups, 
and (2) cross-group super-feature sampling (CSS) to ameliorate long-range relation learning. Specifically, IMM adaptively predicts sparse masks to filter out features with low attention scores 
(\emph{i.e.}, uninformative patterns) during bidirectional Mamba modeling, facilitating the removal of instance redundancies for compact local representation. For improved bag prediction, 
the CSS module further aggregates sparse group representations into discriminative features, effectively grasping comprehensive dependencies among dispersed and sparse tumor regions inherent in large-scale WSIs. 
Extensive experiments on four datasets demonstrate that GMMamba outperforms the state-of-the-art ACMIL by 2.2\% and 6.4\% in accuracy on the TCGA-BRCA and TCGA-ESCA datasets, respectively. 


## Update
- [2025/10/23] Uploading

## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on 3090)

## Dependencies:
```bash
torch
torchvision
numpy
h5py
scipy
scikit-learning
pandas
nystrom_attention
admin_torch
```





The data used for training, validation and testing are expected to be organized as follows:
```bash
DATA_ROOT_DIR/
    ├──DATASET_1_DATA_DIR/
        └── pt_files
                ├── slide_1.pt
                ├── slide_2.pt
                └── ...
        └── h5_files
                ├── slide_1.h5
                ├── slide_2.h5
                └── ...
    ├──DATASET_2_DATA_DIR/
        └── pt_files
                ├── slide_a.pt
                ├── slide_b.pt
                └── ...
        └── h5_files
                ├── slide_i.h5
                ├── slide_ii.h5
                └── ...
    └── ...
```
