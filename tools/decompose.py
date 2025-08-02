import numpy as np
from PIL import Image
from io import BytesIO

import torch
import os
from tqdm import tqdm


def convert_images(image):

    image = Image.open(BytesIO(image.numpy().tobytes()))
    return np.array(image)


def save_each(data_item, output_dir):
    key = data_item["key"]
    torch.save(data_item, os.path.join(output_dir, key+".torch"))

invalid_keys = []

def process_chunk(chunk, output_dir):
    data = torch.load(chunk)

    for data_item in data:
        images = data_item["images"]
        image = convert_images(images[0])
        if image.shape != (360, 640, 3):
            invalid_keys.append(data_item["key"])
            print(f"Skipping {data_item['key']} because the image shape is {image.shape}")
            continue
        save_each(data_item, output_dir)


def main(split):
    data_dir = '../../re10k'
    output_dir = '../../re10k_decomposed/' + split
    os.makedirs(output_dir, exist_ok=True)
    chunks = []

    data_dir = os.path.join(data_dir, split)

    for file in os.listdir(data_dir):
        if file.endswith(".torch"):
            chunks.append(os.path.join(data_dir, file))


    for chunk in tqdm(chunks):
        process_chunk(chunk, output_dir)

    # save invalid keys
    with open(os.path.join(output_dir, "invalid_keys.txt"), "w") as f:
        for key in invalid_keys:
            f.write(key + "\n")

if __name__ == "__main__":
    main("train")
