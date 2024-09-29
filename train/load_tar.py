# import tarfile
# from io import BytesIO
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms

# class TarImageDataset(Dataset):
#     def __init__(self, tar_path, transform=None):
#         self.tar_path = tar_path
#         self.transform = transform
#         self.image_files = []

#         # 读取tar文件并列出所有图片文件
#         with tarfile.open(tar_path, 'r') as tar:
#             for member in tar.getmembers():
#                 if member.name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     self.image_files.append(member.name)

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         with tarfile.open(self.tar_path, 'r') as tar:
#             image_file = tar.extractfile(self.image_files[idx])
#             image_data = image_file.read()
#             image = Image.open(BytesIO(image_data))

#         if self.transform:
#             image = self.transform(image)

#         return image

# # 定义图像转换
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # 创建dataset
# dataset = TarImageDataset('/gpfs/public/vl/gjs/dataset/synthetic-dataset-1m-dalle3-high-quality-captions/data-000000.tar', transform=transform)

# # 使用DataLoader加载数据
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# # 使用示例
# for batch in dataloader:
#     # 在这里处理你的批次数据
#     print(batch.shape)
# %matplotlib inline
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn
from random import randrange
import os
os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"

import webdataset as wds
url = '/gpfs/public/vl/gjs/dataset/synthetic-dataset-1m-dalle3-high-quality-captions/data-000000.tar'
pil_dataset = wds.WebDataset(url).shuffle(1000).decode("pil").to_tuple("jpg", "json")

for image, json in pil_dataset:
    break
plt.imshow(image)