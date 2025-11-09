
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader
from datamodule.av2_dataset import Av2Dataset, collate_fn
from pathlib import Path


class Av2DataModule(LightningDataModule):
    def __init__(
            self,
            data_root: str,
            train_batch_size: int = 32,
            val_batch_size: int = 32,
            shuffle: bool = True,
            num_woker: int = 8,
            pin_memory: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_woker = num_woker
        self.pin_memory = pin_memory

    def setup(self, stage):
        self.train_dataset = Av2Dataset(
            data_root=self.data_root, cached_split="train")
        self.val_dataset = Av2Dataset(
            data_root=self.data_root, cached_split="val")
        self.test_dataset = Av2Dataset(
            data_root=self.data_root, cached_split="test"
        )

    def train_dataloader(self):
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_woker,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn)

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_woker,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn)

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size, 
            shuffle=False,              
            num_workers=self.num_woker,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
