import sys
sys.path.append('../..')
from model import longclip
import torch
from torchvision.datasets import CocoCaptions
from PIL import Image
import numpy as np

import pandas as pd
import json
import os
from tqdm import tqdm
import open_clip
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _generate_examples(examples_json):
    """Yields examples."""
    images = []
    captions = []
    with open(examples_json, 'r') as f:
        dataset = json.load(f)
        for r in dataset:
            images.append(r['image'])
            captions.append(r['conversations'][1]["value"])
    return images, captions

images, captions = _generate_examples("/ML-A100/team/mm/gujiasheng/Long-CLIP/eval/retrieval/gpt4v_api_chat_240406.034614_onlyLocalPath_long.json")


def get_text_feature():
    text_list = []
    feature_list = []
    text_list = captions
    len_list = len(text_list)
    #avoid OOM
    chunks = 50
    with torch.no_grad():
        for i in tqdm(range(chunks)):
            text = text_list[i*len_list//chunks: (i+1)*len_list//chunks]
            text = tokenizer(text).to(device)
            feature_list.append(model.encode_text(text).to('cpu'))
    
    
    text_feature = torch.concatenate(feature_list, dim=0)
    return text_feature
    

def get_image_feature():
    img_feature_list = []
    with torch.no_grad():
        for image_name in tqdm(images):
            image = Image.open(image_name)
            image = preprocess(image).unsqueeze(0).to(device)
            img_feature = model.encode_image(image).to('cpu')
            img_feature_list.append(img_feature)
            torch.cuda.empty_cache()
            del img_feature, image

    img_feature = torch.concatenate(img_feature_list, dim=0)
    return img_feature

def get_accuracy_t2i(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0

        sim = (text_feature @ image_feature.T).softmax(dim=-1)

        for i in range(text_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            if i in topk:
                pred_true = pred_true + 1

        print(pred_true/text_feature.shape[0])

def get_accuracy_i2t(text_feature, image_feature, k):
    with torch.no_grad():
    
        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)



        pred_true = 0

        sim = (image_feature @ text_feature.T).softmax(dim=-1)
        for i in range(image_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            if i in topk:
                pred_true = pred_true + 1

        print(pred_true/image_feature.shape[0])

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    #             "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device
    #         )
    # preprocess = preprocess_val
    # tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    model_path = "/ML-A100/team/mm/gujiasheng/Long-CLIP/train/lr=1e-06_wd=0.01_wl=200_log_scale=4.6052_bs=128/ckpt/longclip-bigG_epoch_5.pt"
    model, preprocess = longclip.load(model_path, device=device)
    tokenizer = longclip.tokenize
    model.eval()
    model_name =  model_path.split("/")[-1].split(".")[0]
    if os.path.exists(f"text_feature_{model_name}.npy"):
        text_feature = np.load(f"text_feature_{model_name}.npy")
    else:
        text_feature = get_text_feature()
        np.save(f"text_feature_{model_name}.npy", text_feature)
    if os.path.exists(f"image_feature_{model_name}.npy"):
        image_feature = np.load(f"image_feature_{model_name}.npy")
    else:
        image_feature = get_image_feature()
        np.save(f"image_feature_{model_name}.npy", image_feature)
    text_feature = torch.tensor(text_feature)
    image_feature = torch.tensor(image_feature)
    get_accuracy_i2t(text_feature, image_feature, 1)
    get_accuracy_i2t(text_feature, image_feature, 5)
    get_accuracy_i2t(text_feature, image_feature, 10)
    get_accuracy_t2i(text_feature, image_feature, 1)
    get_accuracy_t2i(text_feature, image_feature, 5)
    get_accuracy_t2i(text_feature, image_feature, 10)

# bigg
# 0.8922143780451427
# 0.9787212886546683
# 0.9896589440190912
# 0.8070995326638163
# 0.940041761956846
# 0.9651983692950183

# 0.899174704186139
# 0.9839912498757084
# 0.992641940936661

# 0.8967883066520831
# 0.9805110868052103
# 0.9898578104802626