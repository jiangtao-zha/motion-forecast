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
        num_woker=cfg.data.num_woker, # 注意拼写
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
        dropout=cfg.model.dropout, 
        activation=cfg.model.activation,
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay
    )
    

    # --- 3. 配置 Logger ---
    # 可以根据 cfg 中的参数配置 logger
    logger = TensorBoardLogger("tb_logs", name=cfg.run_name)
    # logger = WandbLogger(project=cfg.project_name, name=cfg.run_name)


    # --- 5. 实例化 Trainer ---
    # 可以直接从 cfg 初始化大部分参数
    trainer = pl.Trainer(
        **cfg.trainer, # 使用 trainer 配置组下的所有参数
        logger=logger,
        # callbacks=[checkpoint_callback, early_stop_callback, lr_monitor]
    )

    # --- 运行 LR Finder ---
    print("Finding optimal learning rate...")
    tuner = pl.tuner.Tuner(trainer)
    lr_finder_result = tuner.lr_find(model_module, datamodule=data_module)

    # 绘制建议图表
    fig = lr_finder_result.plot(suggest=True)
    fig.savefig("lr_finder_plot.png")

    # 获取建议值并更新配置
    suggested_lr = lr_finder_result.suggestion()
    print(f"Suggested LR: {suggested_lr}")
    # model_module.hparams.learning_rate = suggested_lr 
    # 或者 cfg.optim.learning_rate = suggested_lr

    # tuner.scale_batch_size(model_module, datamodule=data_module, mode='binsearch')

    # --- 6. 开始训练 ---
    # print("Starting training...")
    # trainer.fit(model_module, datamodule=data_module) # 使用 datamodule 关键字参数



if __name__ == '__main__':
    main()