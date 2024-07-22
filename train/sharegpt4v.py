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


data4v_root = "sharegpt4v/data/"
json_name = "share-captioner_coco_lcs_sam_1246k_1107.json"
image_root = "sharegpt4v/data/"


class share4v_val_dataset(data.Dataset):
    def __init__(self, batch_size=64, num_processes=1, preprocess=None):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        self.batch_size = batch_size
        self.num_processes = num_processes
        with open(data4v_root + json_name, "r", encoding="utf8") as fp:
            self.json_data = json.load(fp)[: self.total_len]
        self.preprocess = preprocess
        # if (
        #     model_name
        #     == "/ML-A100/team/mm/gujiasheng/Long-CLIP/ViT-bigG-14-laion2b_s39b_b160k.pt"
        # ):
        #     (
        #         model,
        #         preprocess_train,
        #         preprocess_val,
        #     ) = open_clip.create_model_and_transforms(
        #         "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
        #     )
        #     preprocess = preprocess_val
        #     del model
        # elif model_name == "ViT-L/14":
        #     model, preprocess = clip.load("ViT-L/14")
        #     del model
        # else:
        #     preprocess = None
        # self.preprocess = preprocess
        # torch.cuda.empty_cache()

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]["conversations"][1]["value"]
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]["image"]
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption, index % (self.batch_size * self.num_processes)


class share4v_train_dataset(data.Dataset):
    def __init__(self, batch_size=64, num_processes=1, preprocess=None):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        self.batch_size = batch_size
        self.num_processes = num_processes
        with open(data4v_root + json_name, "r", encoding="utf8") as fp:
            self.json_data = json.load(fp)[self.total_len :]
        self.preprocess = preprocess
        # if (
        #     model_name
        #     == "/ML-A100/team/mm/gujiasheng/Long-CLIP/ViT-bigG-14-laion2b_s39b_b160k.pt"
        # ):
        #     (
        #         model,
        #         preprocess_train,
        #         preprocess_val,
        #     ) = open_clip.create_model_and_transforms(
        #         "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
        #     )
        #     preprocess = preprocess_val
        #     del model
        # elif model_name == "ViT-L/14":
        #     model, preprocess = clip.load("ViT-L/14")
        #     del model
        # else:
        #     preprocess = None
        # self.preprocess = preprocess
        # torch.cuda.empty_cache()

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]["conversations"][1]["value"]
        caption = caption.replace("\n", " ")

        caption_short = caption.split(". ")[0]

        image_name = self.image_root + self.json_data[index]["image"]
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return (
            image_tensor,
            caption,
            caption_short,
            index % (self.batch_size * self.num_processes),
        )
