import json
import cv2
from PIL import Image
import torch
import torch.utils.data as data
import os
import numpy as np
import random
import glob
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
data_dir = (
    "/gpfs/public/vl/gjs/dataset/synthetic-dataset-1m-dalle3-high-quality-captions"
)
# json_name = "/ML-A100/team/mm/gujiasheng/Long-CLIP/paths_to_save.json"
def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class share4v_train_dataset(data.Dataset):
    def __init__(self, preprocess=None, use_embed=False, start_idx=0, end_idx=32000):
        self.data_dir = data_dir
        self.image_paths = []
        for path in glob.glob(data_dir + "/*"):
            if path.endswith(".jpg") or path.endswith(".png") or path.endswith(".jpeg"):
                self.image_paths.append(path)
        self.image_paths = self.image_paths[start_idx:end_idx]
        self.preprocess = preprocess
        self.use_embed = use_embed

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        try:
            file_name, image_ext = os.path.splitext(self.image_paths[index])
            json_path = file_name + '.json'
            with open(json_path, "r", encoding="utf8") as fp:
                data = json.load(fp)

            if random.random() <= 0.95:
                caption = data["long_caption"]
            else:
                caption = data["short_caption"]
            caption = caption.replace("\n", " ")
            caption_short = caption.split(". ")[0]
            img_path = self.image_paths[index]
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
            return self.__getitem__(random.randint(0, len(self.image_paths) - 1))

testset = share4v_train_dataset(
    preprocess=_transform(224),
    start_idx=0,
    end_idx=4000,
)