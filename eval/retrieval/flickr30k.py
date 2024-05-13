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

_INPUT_IMAGES = "flickr30k-images"
_JSON_KEYS = ['raw', 'sentids']

def _generate_examples(examples_csv, images_dir):
    """Yields examples."""
    df = pd.read_csv(examples_csv)
    for c in _JSON_KEYS:
        df[c] = df[c].apply(json.loads)
    images = []
    captions = []
    for r_idx, r in df.iterrows():
        r_dict = r.to_dict()
        image_path = os.path.join(images_dir, _INPUT_IMAGES, r_dict['filename'])
        r_dict['image'] = image_path
        r_dict['caption'] = r_dict.pop('raw')
        images.append(r_dict['image'])
        captions.extend(r_dict['caption'])
    return images, captions

images, captions = _generate_examples("/ML-A100/team/mm/gujiasheng/Long-CLIP/eval/retrieval/flickr_annotations_30k.csv", "/ML-A100/team/mm/gujiasheng/Long-CLIP/eval/retrieval/")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device
        )
preprocess = preprocess_val
tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
def get_text_feature():
    text_list = []
    feature_list = []
    # with torch.no_grad():
    #     with open("data/flickr/results_20130124.token", 'r') as f:
    #         dataset = f.readlines()
    #         for data in dataset:
    #             image = data.split('\t')[0]
    #             text = data.split('\t')[1]
    #             text_list.append(text)
    #     len_list = len(text_list)
    #     print(len_list)
    text_list = captions
    len_list = len(text_list)
    #avoid OOM
    chunks = 50
    with torch.no_grad():
        for i in tqdm(range(chunks)):
            text = text_list[i*len_list//chunks: (i+1)*len_list//chunks]
            # text = longclip.tokenize(text, truncate=True).to(device)
            text = tokenizer(text).to(device)
            feature_list.append(model.encode_text(text).to('cpu'))
    
    
    text_feature = torch.concatenate(feature_list, dim=0)
    return text_feature
    

def get_image_feature():
    text_list = []
    data_root = "data/flickr/flickr30k-images/"
    img_feature_list = []
    with torch.no_grad():
        # with open("data/flickr/results_20130124.token", 'r') as f:
        #     dataset = f.readlines()
        #     data_len = len(dataset)
            # for i in range(data_len//5):
            #     #1 image corresponding to 5 captions
            #     data = dataset[5*i]
            #     image_name = data.split('\t')[0][:-2]
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
            true_index = i//5
            if true_index in topk:
                pred_true = pred_true + 1

        print(pred_true/text_feature.shape[0])

def get_accuracy_i2t(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0

        sim = (image_feature @ text_feature.T).softmax(dim=-1)
        for i in range(image_feature.shape[0]):
            pred = sim[i]
            values, topk = pred.topk(k)
            for j in range(5):
                true_index = 5*i + j
                if true_index in topk:
                    pred_true = pred_true + 1
                    break

        print(pred_true/image_feature.shape[0])

if __name__ == "__main__":
    
    # model, preprocess = longclip.load("/ML-A100/team/mm/gujiasheng/Long-CLIP/longclip-L.pt", device=device)

    model.eval()

    text_feature = get_text_feature()
    np.save("text_feature.npy", text_feature)
    image_feature = get_image_feature()
    np.save("image_feature.npy", image_feature)
    # text_feature = np.load("text_feature.npy")
    # image_feature = np.load("image_feature.npy")
    # text_feature = torch.tensor(text_feature)
    # image_feature = torch.tensor(image_feature)
    get_accuracy_i2t(text_feature, image_feature, 1)
    get_accuracy_i2t(text_feature, image_feature, 5)
    get_accuracy_i2t(text_feature, image_feature, 10)
    get_accuracy_t2i(text_feature, image_feature, 1)
    get_accuracy_t2i(text_feature, image_feature, 5)
    get_accuracy_t2i(text_feature, image_feature, 10)
# our clip bigg
# 0.5719997420519766
# 0.811117559811698
# 0.8797962210614562
# 0.4438963048945637
# 0.6730637776488038
# 0.7533436512542723

# original clip bigg
# 0.6277810021280712
# 0.8495518153092152
# 0.908009286128845
# 0.4641774682401496
# 0.6872767137421809
# 0.7652866447410847

# longclip L
# 0.4859418327207068
# 0.7371509640807378
# 0.8205649061714064
# 0.3967949958083446
# 0.6289095247307668
# 0.7161282001676662

# our longclip 
# 0.5550719030115432
# 0.7968014445089314
# 0.8715418843103114
# 0.4280389501515445
# 0.6589088798607081
# 0.7425678725736764