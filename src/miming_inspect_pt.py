import torch
import argparse
import sys

def inspect_pt(file_path):
    print(f"--- Inspecting: {file_path} ---")
    try:
        # map_location='cpu' 防止报错显存不足或设备不匹配
        data = torch.load(file_path, map_location='cpu',weights_only=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. 检查数据类型
    print(f"Data Type: {type(data)}")

    # 2. 如果是 Tensor，检查详细信息
    if isinstance(data, torch.Tensor):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        
        # 检查是否包含 NaN 或 Inf
        if torch.isnan(data).any():
            print("\033[91mWARNING: Data contains NaNs!\033[0m")
        else:
            print("Check: No NaNs found.")
            
        if torch.isinf(data).any():
            print("\033[91mWARNING: Data contains Infs!\033[0m")
            
        # 打印统计值，看特征是否“活着” (不能全是0)
        print(f"Mean: {data.mean().item():.4f}")
        print(f"Std : {data.std().item():.4f}")
        print(f"Min : {data.min().item():.4f}")
        print(f"Max : {data.max().item():.4f}")
        
        # print("\nFirst 5 rows preview:")
        # print(data[:5])

    # 3. 如果是 List 或 Dict (防备你存错了格式)
    elif isinstance(data, list):
        print(f"It is a List with length: {len(data)}")
        print(f"First element type: {type(data[0])}")
    elif isinstance(data, dict):
        print(f"It is a Dict. Keys: {data.keys()}")
    else:
        print("Unknown data format.")

if __name__ == "__main__":

    inspect_pt("/home/ubuntu/DISK2/ZJT/sept/src/datamiming/1224_val_latent_features.pt")