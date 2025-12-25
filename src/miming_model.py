# train.py
import hydra
from omegaconf import DictConfig, OmegaConf # DictConfig 用于类型提示
import pytorch_lightning as pl
from SEPT_LightningModule import SEPT_Module
from datamodule.av2_datamodule import Av2DataModule
import torch
import pandas as pd
import os
import pickle 
import numpy as np
torch.set_float32_matmul_precision('medium')

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # cfg 对象现在包含了所有合并后的配置！
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg)) # 打印加载的配置，方便调试

    # --- 0. 设置随机种子 (可选但推荐) ---
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # --- 1. 实例化 DataModule ---
    # Hydra 可以自动实例化 (_target_)，但手动实例化更清晰
    data_module = Av2DataModule(
        data_root=cfg.data.data_root,
        train_batch_size=cfg.data.train_batch_size,
        val_batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_woker=cfg.data.num_woker,
        pin_memory=cfg.data.pin_memory
    )

    

    model_module = SEPT_Module.load_from_checkpoint(cfg.ckpt_path)
    trainer = pl.Trainer(
        **cfg.trainer,
        inference_mode=True,
        logger=False,        
    )
    
    print("Starting Inference...")
    data_module.setup(stage="predict")
    predictions = trainer.predict(model_module,dataloaders=data_module.train_dataloader()) 

    print("Processing results...")
    all_data = {
        'id': [], 'entropy': [], 'top1_fde': [], 'is_miss': []
    }
    latent_features = []
    y_hat = []


    for batch_out in predictions:

        if not batch_out: continue
        
        all_data['id'].extend(batch_out['scenario_id'])
        all_data['entropy'].extend(batch_out['entropy'].tolist())
        all_data['top1_fde'].extend(batch_out['top1_fde'].tolist())
        all_data['is_miss'].extend(batch_out['is_miss'].tolist())
            
        latent_features.append(batch_out['latent_feat']) 
        y_hat.append(batch_out['y_hat'])

    # --- 6. 保存 ---
    output_dir = "/home/ubuntu/DISK2/ZJT/sept/src/datamiming"
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(all_data)
    # 按 FDE 降序排列，方便直接查看最难的
    df = df.sort_values(by='top1_fde', ascending=False)
    
    # 保存 CSV
    csv_path = os.path.join(output_dir, "mining_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

    # 保存特征
    feat_path = os.path.join(output_dir, "1224_val_latent_features.pt")
    torch.save(torch.cat(latent_features, dim=0), feat_path)
    print(f"Latent features saved to {feat_path}")

    # 保存为字典 {scenario_id: numpy_array} (强烈推荐，用于可视化)
    print("Converting predictions to dictionary...")
    all_preds_tensor = torch.cat(y_hat, dim=0)
    # 确保 ID 列表和预测 Tensor 长度一致
    assert len(all_data['id']) == len(all_preds_tensor), "ID数量与预测样本数不一致！"

    pred_dict = {}
    # 将 Tensor 转为 Numpy 并存入字典，减小体积且通用性更好
    for sid, pred in zip(all_data['id'], all_preds_tensor):
        pred_dict[sid] = pred.detach().cpu().numpy()

    # 使用 pickle 保存字典
    pred_dict_path = os.path.join(output_dir, "1224_val_predictions_dict.pkl")
    with open(pred_dict_path, 'wb') as f:
        pickle.dump(pred_dict, f)
        
    print(f"Predictions dictionary saved to {pred_dict_path}")

if __name__ == '__main__':
    main()