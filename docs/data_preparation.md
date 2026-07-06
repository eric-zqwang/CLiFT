## Data preparation
We use [RealEstate10K](https://google.github.io/realestate10k/index.html) and [DL3DV](https://github.com/DL3DV-10K/Dataset) datasets.

### RealEstate10K
For the RE10K dataset, we follow the [PixelSpalt](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) to get the data. After downloading the data, you can use [this script](../tools/decompose.py) to convert to our data format.

For second-stage training, you'll also need to download the pre-computed K-means assignments to enable fast training. These can be downloaded from [Hugging Face](https://huggingface.co/EricW123456/CLiFT/blob/main/re10k/kmeans_faiss_no_features_merged.zip) (or generated from your first-stage checkpoint with `script/train/re10k/save_kmeans.sh`).

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
We use the official [DL3DV-10K download script](https://github.com/DL3DV-10K/Dataset) to fetch the 480P images and poses (no further conversion is needed):

```
python scripts/download.py --odir DL3DV-ALL-480P-images --subset ${subset} --resolution 480P --file_type images+poses --clean_cache
```

Each scene directory must contain a `transforms.json` (camera poses + intrinsics) and an `images_8/` folder with `frame_*.png` frames at 270x480. Point `data_dir` in `config/data/dl3dv.yaml` to the download folder (default: `../Dataset/DL3DV-ALL-480P-images/`):

```
../Dataset/
└── DL3DV-ALL-480P-images
    ├── 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7
    │   ├── transforms.json
    │   └── images_8
    │       ├── frame_00001.png
    │       ├── ...
    ├── ...
```

For second-stage (condenser) training you additionally need per-scene K-means assignments under `../Dataset/dl3dv_kmeans_faiss_merged/`. Download them from [Hugging Face](https://huggingface.co/EricW123456/CLiFT/blob/main/dl3dv/dl3dv_kmeans_faiss_merged.tar.zst) and extract:

```
tar --zstd -xf dl3dv_kmeans_faiss_merged.tar.zst -C ../Dataset/
```

or generate them from your first-stage checkpoint with `script/train/dl3dv/save_kmeans.sh` (see the [training guide](training.md)).