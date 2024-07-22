from datetime import datetime
import time
import json
import os
import threading
from queue import Queue
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import sys

from model import longclip


class ProgressCounter:
    def __init__(self, total_items, gpu_id):
        self.total_items = total_items
        self.processed_items = 0
        self.gpu_id = gpu_id
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.processed_items += 1
            progress = self.processed_items / self.total_items * 100
            print(
                f"Progress {self.gpu_id}: {progress:.2f}% ({self.processed_items}/{self.total_items} items)",
                end="\r",
            )


def extract_caption_vit(json_fps, gpu_id):
    print(f"[GPU{gpu_id}] Start time:", datetime.now())
    t0 = time.time()

    global mutex
    mutex = threading.Lock()
    jobs = Queue()

    for item in tqdm(json_fps, desc="put json label paths"):
        jobs.put(item)

    worker_num = 50
    print(f"[GPU{gpu_id}] worker number: {worker_num}")
    progress_counter = ProgressCounter(len(json_fps), gpu_id=gpu_id)
    for _ in range(worker_num):
        worker = threading.Thread(
            target=extract_caption_vit_do, args=(jobs, progress_counter)
        )
        worker.start()

    jobs.join()

    print("[GPU{gpu_id}] End time:", datetime.now())
    print(f"[GPU{gpu_id}] cost {(time.time() - t0) / 3600} hours")


def extract_caption_vit_do(q, progress_counter):
    while not q.empty():
        item = q.get()
        extract_caption_vit_job(item)
        progress_counter.increment()
        q.task_done()


def extract_caption_vit_job(item):
    """item is JSON label path"""
    if not os.path.exists(item):
        print(f"label path {item} not exists.")
        return

    global mutex
    global vit

    with torch.no_grad():
        try:
            json_content = json.load(open(item, "r"))
        except:
            print(f"failed to load {item}")
            return

        assert "/json/" in item and ".json" in item
        save_path = item.replace("/json/", "/vit_emb/").replace(".json", "")
        if os.path.exists(save_path):
            print(f"{save_path} already exists.")
            return

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        try:
            mutex.acquire()

            transform = longclip._transform(vit.input_resolution)
            img_path = json_content["img_path"]
            size = os.path.getsize(img_path)
            if size > 24 * 1e6:  # filter very big images
                print(
                    f"Skipping image due to excessive size: {size} bytes. Path: '{img_path}'."
                )
                res = None
            else:
                image = Image.open(img_path)
                if image.mode == "P":
                    image = image.convert("RGBA").convert("RGB")
                elif image.mode != "RGB":
                    image = image.convert("RGB")
                image_tensor = (
                    transform(image).to(next(vit.parameters()).device).unsqueeze(0)
                )
                res = vit(image_tensor)  # 'embedding', 'text', 'length'

            mutex.release()
            if res is not None:
                np.save(save_path, res.cpu().numpy())
        except Exception as e:
            print(f"when processing {item}, got {e}")


import torch.multiprocessing as mp


def split_file_indices(file_cnt, task_num):
    files_per_task = file_cnt // task_num
    extra_files = file_cnt % task_num

    indices = []
    start = 0

    for i in range(task_num):
        end = start + files_per_task + (1 if i < extra_files else 0)
        indices.append((start, end))
        start = end

    return indices


def run_task_on_gpu(gpu_id, json_fps):
    global vit
    vit = torch.load("ViT-bigG-14.pth", map_location=f"cuda:{gpu_id}")
    print(f"ViT model loaded on GPU {gpu_id}")
    extract_caption_vit(json_fps, gpu_id)


if __name__ == "__main__":
    """read label files from {dataset_root}/json, and save vit embeddings into {dataset_root}/vit_emb
    /ML-A100/team/mm/wangjiazhi/.cache/IF_/vit-v1_1-xxl
    should proxy_off before run this script
    conda activate pixart  (/ML-A100/team/mm/wangjiazhi/anaconda3/envs/pixart/bin/python)
    single process: speed (batch=1, no token padding): 22.75 images/sec/GPU (each image has two captions to be encoded.)
    3 processes: 65 images/sec/3GPU
    8 processes: 140 images/sec/8GPU
    """

    json_name = "/ML-A100/team/mm/gujiasheng/Long-CLIP/paths_not_exists.json"
    with open(json_name, "r", encoding="utf8") as fp:
        json_label_paths = json.load(fp)
    json_label_paths = json_label_paths
    available_gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    # available_gpu_ids = [0,1,2,3]
    # available_gpu_ids = [1,2,3]
    # available_gpu_ids = [0]
    print(f"To process {json_name} with GPU {available_gpu_ids}")

    index_ranges = split_file_indices(len(json_label_paths), len(available_gpu_ids))
    assert len(index_ranges) == len(available_gpu_ids)

    json_label_chunks = []
    for i, (start, end) in enumerate(index_ranges):
        json_label_paths_part = json_label_paths[start:end]
        print(
            f"part {i} [GPU{available_gpu_ids[i]}]: {len(json_label_paths_part)} json files, from {start} to {end-1}"
        )
        json_label_chunks.append(json_label_paths_part)

    mp.set_start_method("spawn", force=True)
    processes = []
    for i in range(len(available_gpu_ids)):
        p = mp.Process(
            target=run_task_on_gpu, args=(available_gpu_ids[i], json_label_chunks[i])
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
