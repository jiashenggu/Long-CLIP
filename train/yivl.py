import json
import cv2
from PIL import Image
import clip

import torch
import torch.utils.data as data
import os
import numpy as np
import random
import open_clip


json_name = "/ML-A100/team/mm/gujiasheng/Long-CLIP/train/yivl_data_paths.json"
# json_name = "/ML-A100/team/mm/gujiasheng/Long-CLIP/paths_to_save.json"


class share4v_val_dataset(data.Dataset):
    def __init__(self, preprocess=None, use_embed=False):
        self.json_name = json_name
        self.total_len = 32000
        with open(json_name, "r", encoding="utf8") as fp:
            self.json_data = json.load(fp)[: self.total_len]
        self.preprocess = preprocess
        self.use_embed = use_embed

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        try:
            with open(self.json_data[index], "r", encoding="utf8") as fp:
                data = json.load(fp)

            if random.random() <= 0.95:
                if "yivl2" in data:
                    caption = data["yivl2"]
                else:
                    caption = data["yivl"]
            else:
                caption = data["clean_prompt"]
            caption = caption.replace("\n", " ")
            caption_short = caption.split(". ")[0]
            img_path = data["img_path"]
            if self.use_embed:
                image_embed_name = img_path.replace("/json/", "/vit_emb/").replace(
                    ".json", ".npy"
                )
                image_tensor = np.load(image_embed_name)
            else:
                image = Image.open(img_path)
                image_tensor = self.preprocess(image)
            return (
                image_tensor,
                caption,
                caption_short,
            )
        except Exception as e:
            print("dataset exception", e)
            return self.__getitem__(random.randint(0, len(self.json_data) - 1))


class share4v_train_dataset(data.Dataset):
    def __init__(self, preprocess=None, use_embed=False):
        self.json_name = json_name
        self.total_len = 32000
        with open(json_name, "r", encoding="utf8") as fp:
            self.json_data = json.load(fp)[self.total_len :]
        self.preprocess = preprocess
        self.use_embed = use_embed

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        try:
            with open(self.json_data[index], "r", encoding="utf8") as fp:
                data = json.load(fp)

            # if random.random() <= 0.95:
            #     if "yivl2" in data:
            #         caption = data["yivl2"]
            #     else:
            #         caption = data["yivl"]
            # else:
            #     caption = data["clean_prompt"]
            if "yivl2" in data:
                caption = data["yivl2"]
            else:
                caption = data["yivl"]

            if not isinstance(caption, str):
                raise Exception("caption is not string")
            caption = caption.replace("\n", " ")
            caption_short = caption.split(". ")[0]
            img_path = data["img_path"]
            img_path_ord = [ord(char) for char in img_path]
            img_path_ord.extend([-1] * (512 - len(img_path_ord)))
            if self.use_embed:
                image_embed_name = img_path.replace("/json/", "/vit_emb/").replace(
                    ".json", ".npy"
                )
                image_tensor = np.load(image_embed_name)
            else:
                image = Image.open(img_path)
                image_tensor = self.preprocess(image)
            return (
                image_tensor,
                caption,
                caption_short,
                torch.tensor(img_path_ord, dtype=torch.int),
            )
        except Exception as e:
            print("dataset exception", e)
            return self.__getitem__(random.randint(0, len(self.json_data) - 1))
