# test_dataloader.py

import torch
from pathlib import Path
from av2_datamodule import Av2DataModule
torch.set_printoptions(profile="full", threshold=100000, linewidth=100)
def main():
    # --- 1. 配置您的数据路径 ---
    # 请将此路径修改为您存储预处理数据(.pt文件)的根目录
    # 例如：如果您的数据在 '.../processed_data/train' 和 '.../processed_data/val'
    # 那么 DATA_ROOT 应设置为 '.../processed_data'
    DATA_ROOT = Path("/home/ubuntu/DISK2/ZJT/argoverse_dataset_v2/sept_small/") # 假设您的数据根目录是 ./data
    
    # data_folder参数现在由datamodule内部处理，我们只需要提供根目录
    
    # --- 2. 初始化 Av2DataModule ---
    # 我们可以用一个较小的batch_size进行测试
    print("Initializing Av2DataModule...")
    datamodule = Av2DataModule(
        data_root=DATA_ROOT,
        train_batch_size=32,
        val_batch_size=1,
        num_woker=0  # 在Windows或调试时，建议设为0以避免多进程问题
    )

    # --- 3. 设置并获取 DataLoader ---
    # 调用 .setup() 会初始化 train_dataset 和 val_dataset
    print("Setting up data...")
    datamodule.setup(stage='fit') 
    
    # 获取训练数据加载器
    train_loader = datamodule.train_dataloader()
    print(f"Successfully created DataLoader.")
    print(f"Total number of training samples: {len(datamodule.train_dataset)}")
    print("-" * 50)

    # --- 4. 加载并检查一个批次的数据 ---
    print("Fetching one batch of data to inspect...")
    try:
        # 从DataLoader中获取第一个批次的数据
        batch = next(iter(train_loader))
        
        print("Successfully fetched one batch!")
        print("-" * 50)
        print("Data batch inspection:")
        
        # 遍历批次字典中的每一个键
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # 如果值是Tensor，打印其形状、数据类型和设备
                print(f"  - Key: '{key}'")
                print(f"    - Shape: {value.shape}")
                print(f"    - Dtype: {value.dtype}")
                print(f"    - Device: {value.device}")
            else:
                # 其他类型的值直接打印
                print(f"  - Key: '{key}': {value}")
                
        # 还可以额外检查一些关键计算值
        print("-" * 50)
        print("Additional checks:")
        print(f"Number of actors per sample in batch: {batch['num_actors']}")
        print(f"Number of lanes per sample in batch: {batch['num_lanes']}")
        print(f"Number of x_attr per sample in batch: {batch['x_attr'][0,:,:]}")
        # print(f"Number of lane_src per sample in batch: {batch['lane_src'][0,10:,:]}")
        # print(f"x_padding_mask in batch: {batch['x_padding_mask'][...,:50].all(-1)}")
        # print(f"one_x_padding_mask in batch: {batch['x_padding_mask'][:,0.:]}")
        # print(f"Example of x_diff tensor:\n{batch['x_diff'][:2]}") # 打印前2个样本的原点
        # print(f"打印一条道路：{batch['lane_positions'][1,1]}")

    except StopIteration:
        print("DataLoader is empty. Please check if your data folder is correct and contains .pt files.")
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        print("Please double-check the fix in av2_extractor.py and ensure you have regenerated the .pt files.")


if __name__ == '__main__':
    main()