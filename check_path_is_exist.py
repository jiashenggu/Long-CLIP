import os
from multiprocessing import Pool
import json
def check_path_existence(path):
    """
    Check if a single path exists.
    
    Args:
    path (str): Path to check.
    
    Returns:
    bool: True if the path exists, False otherwise.
    """
    return os.path.exists(path)

def check_paths_exist_parallel(paths):
    """
    Check if a list of paths exists using parallel processing.
    
    Args:
    paths (list): List of paths to check.
    
    Returns:
    list: List of boolean values indicating whether each path exists or not.
    """
    with Pool() as pool:
        existence_status = pool.map(check_path_existence, paths)
    return existence_status


yivl_data_paths = json.load(open("/ML-A100/team/mm/gujiasheng/Long-CLIP/train/yivl_data_paths.json"))
print(len(yivl_data_paths))
paths_to_check = []
for item in yivl_data_paths:
    item = item.replace('/json/', '/vit_emb/').replace('.json', '.npy')
    paths_to_check.append(item)

import time
start = time.time()
existence_status = check_paths_exist_parallel(paths_to_check)
paths_to_save = []
paths_not_exists = []
for path, exists in zip(paths_to_check, existence_status):
    # print(f"Path '{path}' exists: {exists}")
    if exists:
        paths_to_save.append(path)
    else:
        paths_not_exists.append(path.replace('/vit_emb/', '/json/').replace('.npy', '.json'))
print(len(paths_to_save))
print(len(paths_not_exists))
json.dump(paths_to_save, open("paths_to_save.json", "w"))
json.dump(paths_not_exists, open("paths_not_exists.json", "w"))

use_time = (time.time() - start)/len(paths_to_check)
print(f"Time taken: {time.time() - start}")
print(f"Speed: {use_time} seconds/sample")
