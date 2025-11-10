
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Av2Dataset(Dataset):
    def __init__(
            self,
            data_root: Path,
            cached_split: str = None):
        super().__init__()
        if data_root is not None and cached_split is not None:
            self.data_folder = Path(data_root) / cached_split
            self.file_list = sorted(list(self.data_folder.glob("*.pt")))

        else:
            raise ValueError(
                "Av2Dataset Error : data_root or cached_split must be specified")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        return torch.load(self.file_list[index], weights_only=True)


def collate_fn(batch):
    data = {}

    for key in [
        "x_src",
        "lane_src",
        # "x_center",
        # "x_heading",
        # "lane_centers",
        # "lane_angles",
        "agent_pos_feat",
        "road_pos_feat",

        "lane_attr",
        # "lanes_start",  # [batch num_road 2]
        # "lanes_end",  # [batch num_road 2]
        # "lanes_length",  # [batch num_road]
        # "is_intersections",

        # "x_velocity",
        "x_attr",
        # "lane_positions",


        # "x_diff",
    ]:
        data[key] = pad_sequence(
            [b[key] for b in batch], batch_first=True, padding_value=0.0)

    if batch[0]["y_diff"] is not None:
        data["y_diff"] = pad_sequence([b["y_diff"]
                                      for b in batch], batch_first=True, padding_value=0.0)

    for key in ["x_padding_mask", "lane_padding_mask"]:
        data[key] = pad_sequence(
            [b[key] for b in batch], batch_first=True, padding_value=True
        )
    data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)
    data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)

    data["num_actors"] = (~data["x_key_padding_mask"]).sum(-1)
    data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)

    # for name, tensor in data.items():
    #     # 确保只检查 torch.Tensor 且不是布尔型的张量
    #     if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.bool:

    #         # --- 检查 NaN ---
    #         if torch.isnan(tensor).any():
    #             # raise AssertionError 抛出错误并停止执行
    #             raise AssertionError(f"输入张量 '{name}' 中包含 NaN (Not a Number) 值！请检查预处理步骤。")

    #         # --- 检查 Inf ---
    #         if torch.isinf(tensor).any():
    #             # raise AssertionError 抛出错误并停止执行
    #             raise AssertionError(f"输入张量 '{name}' 中包含 Inf (无穷大) 值！这通常是数值溢出导致。")

    return data
