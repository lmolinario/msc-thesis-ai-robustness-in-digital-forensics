# Introduction

This is the Firearm14k dataset we use to train and test our firearm retrieval model. It contains 3 folders, i.e., `train`, `validation` and `test`. 

# Folder structure

Under the folder `train`, there are 107 sub-directories, each containing 1 category of firearm image. The name of each sub-directory is the firearm name. The number of images in each folder varies. 

The structure of `test` and `validation` folder are the same. Under the directory, there are 2 sub-directories, one is `database_image`, the other is `query_info`. The `database_image` directory contains all the database images. Under the folder `query_info`, there is a file named `ground_truth_info.json` and a folder named `query_image`. The file `ground_truth_info.json`  contains a dict, in which key is the name of each query image and value is the correponding ground truth image in the folder `database_image`. All the query images are placed in the folder `query_image`.

In summary, the directory structure for `test` and `validation` will be something like the following (the files has been omitted.)

```
.
├── database_image
└── query_info
    └── query_image

```

# Cite information

If you use our dataset, please cite this paper:

```
@inproceedings{Hao2018DeepFirearm,
  title={DeepFirearm: Learning Discriminative Feature Representation for Fine-grained Firearm Retrieval},
  author={Jiedong Hao, Jing Dong, Wei Wang, Tieniu Tan},
  booktitle={2018 24th International Conference on Pattern Recognition (ICPR)},
  year={2018}
}

```