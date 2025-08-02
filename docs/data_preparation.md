## Data preparation
We use [RealEstate10K](https://google.github.io/realestate10k/index.html) and [DL3DV](https://github.com/DL3DV-10K/Dataset) datasets.

### RealEstate10K
For the RE10K dataset, we follow the [PixelSpalt](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) to get the data. After downloading the data, you can use [this script](../tools/decompose.py) to convert to our data format.

For second-stage training, you'll also need to download the pre-computed K-means assignments to enable fast training. These can be downloaded from [here](https://drive.google.com/file/d/1Tp-_5_WJUnvUFZvRb88OZvcB69tQgGeX/view?usp=sharing).

```
../clift/
└── re10k_data
    ├── re10k_decompossed
    │   ├── train
    │   │   ├── 1157e06fc0b745b3.torch
    |   |   ├── ...
    │   ├── test
    ├── kmeans_faiss_no_features_merged
```



### DL3DV
We will relase the DL3DV related code soon.