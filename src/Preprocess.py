# preprocess.py (已修改为支持多文件夹处理)

import ray
import os
from pathlib import Path
from tqdm import tqdm
import time
import itertools
# 从您提供的文件中导入 Av2Extractor 类
# [重要] 请确保这个导入路径是正确的
from datamodule.Post.av2_extractor import Av2Extractor

# --- 配置区 ---
# [修改] 将原来的具体路径改为基础路径
# 1. 包含 train, val, test 文件夹的基础输入目录
BASE_INPUT_DIR = Path("/home/ubuntu/DISK2/ZJT/argoverse_dataset_v2") 

# 2. 用于存放输出结果的基础目录 (脚本会自动在下面创建 train, val, test 文件夹)
BASE_OUTPUT_DIR = Path("/home/ubuntu/DISK2/ZJT/argoverse_dataset_v2/sept_small")

# 3. [新增] 指定要处理的子文件夹列表
# SPLITS_TO_PROCESS = ["train", "val", "test"]
SPLITS_TO_PROCESS = ["train", "val", "test"]
# 4. 使用的 CPU核心数 (可以设置为 os.cpu_count() 来使用所有核心)
NUM_WORKERS = 25

# 5. Av2Extractor 初始化参数
EXTRACTOR_RADIUS = 150
# ----------------

@ray.remote
class ExtractorActor:
    """
    一个 Ray Actor，封装了 Av2Extractor 实例。
    每个 Actor 都是一个独立的进程，拥有自己的 extractor 对象。
    (此类无需修改)
    """
    def __init__(self, save_path: Path, radius: int):
        # 在 Actor 进程内部初始化 Extractor
        self.extractor = Av2Extractor(save_path=save_path, radius=radius)
        # print(f"Actor (PID: {os.getpid()}) for {save_path.name} initialized.") # 可以取消注释来查看详细信息

    def process_and_save(self, scenario_file: Path):
        """
        调用 extractor 的 save 方法来处理单个文件。
        该方法是远程调用的。
        """
        try:
            # save 方法内部会处理 process 和 torch.save
            self.extractor.save(file=scenario_file)
            return scenario_file.stem, "Success"
        except Exception as e:
            # 捕获处理单个文件时可能出现的任何错误
            return scenario_file.stem, f"Error: {e}"

# [新增] 将核心逻辑封装成一个可重用的函数
def process_split(input_dir: Path, output_dir: Path):
    """处理单个数据集split（如 train, val 或 test）的函数"""
    print(f"\n{'='*20}")
    print(f"Processing split: {input_dir.name}")
    print(f"{'='*20}")
    
    start_time = time.time()

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描输入目录，找到所有 scenario_*.parquet 文件
    print(f"Scanning for scenario files in: {input_dir}")
    # scenario_files = list(input_dir.glob("*/scenario_*.parquet"))
    # scenario_files = scenario_files[:100]
    scenario_files = []
    # 1. 遍历生成器
    for f in input_dir.glob("*/scenario_*.parquet"):
        # 2. 每找到一个就添加到列表中
        scenario_files.append(f)
        # 3. 列表满了100个就立刻停止循环
        # if len(scenario_files) >= 10:
        #     break
    
    if not scenario_files:
        print(f"Warning: No 'scenario_*.parquet' files found in {input_dir}. Skipping this split.")
        return
        
    print(f"Found {len(scenario_files)} scenarios to process for '{input_dir.name}'.")

    # 创建 Actor 池
    actors = [
        ExtractorActor.remote(save_path=output_dir, radius=EXTRACTOR_RADIUS) 
        for _ in range(NUM_WORKERS)
    ]

    # 分发任务
    futures = []
    for i, file_path in enumerate(scenario_files):
        actor = actors[i % NUM_WORKERS]
        futures.append(actor.process_and_save.remote(file_path))

    # 使用 tqdm 显示进度并获取结果
    results = []
    desc = f"Processing {input_dir.name}"
    with tqdm(total=len(futures), desc=desc) as pbar:
        while futures:
            ready, futures = ray.wait(futures, num_returns=min(NUM_WORKERS, len(futures)))
            results.extend(ray.get(ready))
            pbar.update(len(ready))

    # 统计成功和失败的个数
    success_count = sum(1 for _, status in results if status == "Success")
    errors = [res for res in results if res[1] != "Success"]
    
    end_time = time.time()
    print(f"\n--- '{input_dir.name}' Split Processing Complete ---")
    print(f"Total time for this split: {end_time - start_time:.2f} seconds")
    print(f"Successfully processed: {success_count}/{len(scenario_files)}")
    if errors:
        print(f"Failed to process: {len(errors)}/{len(scenario_files)}")
        # 为了避免刷屏，可以选择只打印少量错误信息
        for filename, error_msg in errors[:5]: # 只打印前5个错误
            print(f"  - Example Error in {filename}: {error_msg}")

def main():
    """主执行函数，负责初始化Ray并循环处理所有splits"""
    
    # 初始化 Ray
    # 将 ray.init 和 shutdown 放在 main 函数的最外层，避免重复初始化
    ray.init(num_cpus=NUM_WORKERS)
    print(f"Ray initialized with {NUM_WORKERS} workers.")
    
    try:
        # 循环处理在配置区定义的每个split
        for split in SPLITS_TO_PROCESS:
            input_path = BASE_INPUT_DIR / split
            output_path = BASE_OUTPUT_DIR / split
            
            if not input_path.exists():
                print(f"Warning: Input directory not found: {input_path}. Skipping '{split}'.")
                continue

            process_split(input_dir=input_path, output_dir=output_path)

    finally:
        # 确保 Ray 在程序结束时（即使出错）也能被关闭
        ray.shutdown()
        print("\nAll processing finished. Ray has been shut down.")


if __name__ == "__main__":
    main()