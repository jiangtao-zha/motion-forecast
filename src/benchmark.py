import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import time
import numpy as np
from SEPT_LightningModule import SEPT_Module
from datamodule.av2_datamodule import Av2DataModule

# 辅助函数：将 batch 数据递归移动到 GPU
def move_batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    else:
        return batch

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # --- 1. 设置设备 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on: {device}")

    # --- 2. 准备数据 (只取一个 Batch) ---
    print("Loading DataModule...")
    # 强制设置 batch_size 为 1 以测试单帧延迟 (Latency)
    # 如果你想测试吞吐量 (Throughput)，请保持原有的 batch_size 或在 config 中修改
    test_batch_size = 1
    
    data_module = Av2DataModule(
        data_root=cfg.data.data_root,
        train_batch_size=test_batch_size, # 这里其实不重要，我们只用 val/predict dataloader
        val_batch_size=test_batch_size,
        shuffle=False,
        num_woker=0, # benchmark 时不需要多线程加载干扰
        pin_memory=True
    )
    data_module.setup(stage="predict")
    dataloader = data_module.val_dataloader()
    
    # 获取一个真实的 batch 用于测试
    print("Fetching a sample batch...")
    batch = next(iter(dataloader))
    
    # 将数据移动到 GPU
    batch = move_batch_to_device(batch, device)

    # --- 3. 加载模型 ---
    print("Loading Model...")
    # 如果有 checkpoint，加载权重；否则从头初始化（速度测试只需结构正确，权重不影响速度）
    if cfg.get("ckpt_path") and os.path.exists(cfg.ckpt_path):
        model = SEPT_Module.load_from_checkpoint(cfg.ckpt_path)
        print(f"Loaded weights from {cfg.ckpt_path}")
    else:
        # 按照 train.py 中的方式初始化
        model = SEPT_Module(
            agent_input_dim=cfg.model.agent_input_dim,
            road_input_dim=cfg.model.road_input_dim,
            num_layers_Kt=cfg.model.num_layers_Kt,
            num_layers_Ks=cfg.model.num_layers_Ks,
            num_layers_Kc=cfg.model.num_layers_Kc,
            d_model=cfg.model.d_model,
            num_head_Kt=cfg.model.num_head_Kt,
            num_head_Ks=cfg.model.num_head_Ks,
            num_head_Kc=cfg.model.num_head_Kc,
            num_queries=cfg.model.num_queries,
            mlp_ratio = cfg.model.mlp_ratio,
            qkv_bias=cfg.model.qkv_bias,
            linear_bias=cfg.model.linear_bias,
            drop_path=cfg.model.drop_path,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
            learning_rate=cfg.optim.learning_rate,
            weight_decay=cfg.optim.weight_decay,
            train_batch_size = cfg.data.train_batch_size,
            warmup_steps=cfg.optim.warmup_steps,
            start_lr_ratio = cfg.optim.start_lr_ratio,
            min_learning_rate = cfg.optim.min_learning_rate
        )
        print("Initialized model from config (Random Weights)")

    model.to(device)
    model.eval() # 开启评估模式 (关闭 Dropout/BatchNorm 更新)

    # --- 4. 开始测速 ---
    # 定义循环次数
    iterations = 100
    warmup = 20
    
    print(f"\nStarting Benchmark (Batch Size: {test_batch_size})...")
    print(f"Warmup steps: {warmup}, Measured steps: {iterations}")

    # 用于计时的 CUDA Event
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((iterations, 1))

    with torch.no_grad(): # 禁用梯度计算，节省显存并加速
        # 预热循环
        for _ in range(warmup):
            # 注意：这里需要根据你的模型 forward 接收参数的方式进行调整
            # 假设你的 LightningModule 内部处理了 batch 字典，或者 forward 接收字典
            # 如果 forward 接收的是解包参数，请使用 model(**batch)
            _ = model.model(batch) 
            
        torch.cuda.synchronize() # 等待预热结束

        # 正式测速循环
        for rep in range(iterations):
            starter.record()
            
            # --- 核心推理 ---
            _ = model.model(batch)
            # ---------------
            
            ender.record()
            
            # 等待 GPU 完成
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # 返回毫秒
            timings[rep] = curr_time

    # --- 5. 结果统计 ---
    avg_ms = np.mean(timings)
    std_ms = np.std(timings)
    min_ms = np.min(timings)
    
    # 计算 FPS (每秒处理多少个 Sample)
    # timings 是毫秒，所以 avg_ms / 1000 是秒
    avg_s = avg_ms / 1000
    fps = test_batch_size / avg_s

    print("\n" + "="*40)
    print(f"Model Inference Speed Benchmark Results")
    print("="*40)
    print(f"Batch Size: {test_batch_size}")
    print(f"Avg Latency (Batch): {avg_ms:.4f} ms")
    print(f"Std Dev:             {std_ms:.4f} ms")
    print(f"Min Latency:         {min_ms:.4f} ms")
    print("-" * 40)
    print(f"Throughput (FPS):    {fps:.2f} samples/sec")
    print("="*40)

if __name__ == '__main__':
    # 确保必要的 import 存在 (因为代码是粘贴进去的)
    import os
    main()