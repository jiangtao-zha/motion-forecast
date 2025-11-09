# train.py
import hydra
from omegaconf import DictConfig, OmegaConf # DictConfig 用于类型提示
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger # 或 WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from SEPT_LightningModule import SEPT_Module
from datamodule.av2_datamodule import Av2DataModule
import torch
torch.set_float32_matmul_precision('medium')
# --- 使用 Hydra 装饰器 ---
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
        shuffle=cfg.data.shuffle,
        num_woker=cfg.data.num_woker,
        pin_memory=cfg.data.pin_memory
    )

    # --- 2. 实例化 LightningModule ---
    model_module = SEPT_Module(
        # --- 模型参数 ---
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
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout, # 假设 LightningModule 内部使用 drop_out
        activation=cfg.model.activation,
        # --- 训练参数 (优化器和损失权重) ---
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        train_batch_size = cfg.data.train_batch_size,
        warmup_steps=cfg.optim.warmup_steps
    )
    # print(model_module)
    

    # --- 3. 配置 Logger ---
    # 可以根据 cfg 中的参数配置 logger
    logger = TensorBoardLogger("tb_logs", name=cfg.run_name)
    # logger = WandbLogger(project=cfg.project_name, name=cfg.run_name)

    # --- 4. 配置 Callbacks ---
    # 可以硬编码或从 cfg 加载
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', # 监控验证集损失
        dirpath=f'outputs/{cfg.run_name}/checkpoints', # Hydra 会自动管理工作目录
        filename='sept-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       patience=15,
       mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    print("Starting training...")
    trainer = pl.Trainer(
        **cfg.trainer, # 使用 trainer 配置组下的所有参数
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor]
    )
    trainer.fit(model_module,
                datamodule=data_module) # 使用 datamodule 关键字参数
    
    # resume_ckpt_path = "/home/ubuntu/DISK2/ZJT/sept/src/outputs/default_run_2025-11-04_20-08-54/checkpoints/sept-epoch=35-val_loss=2.21.ckpt"
    # trainer.fit(model_module,
    #             datamodule=data_module,
    #             ckpt_path = resume_ckpt_path) # 使用 datamodule 关键字参数

    # print("Starting testing...")
    # trainer = pl.Trainer(
    #     devices=1, num_nodes=1,max_epochs=1,logger=False
    # )
    # trainer.test(
    #     model_module,
    #     datamodule=data_module,
    #     ckpt_path='/home/ubuntu/DISK2/ZJT/sept/src/outputs/default_run_2025-11-09_17-26-47/checkpoints/sept-epoch=17-val_loss=2.48.ckpt' 
    # )
    


if __name__ == '__main__':
    main()