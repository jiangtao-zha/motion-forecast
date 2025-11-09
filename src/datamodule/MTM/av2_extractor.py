# 完成论文中的数据提取，以及数据预处理

from sympy import rad
from ..av2_data_utils import load_av2_df
import torch
import numpy as np


class Av2Extractor:
    def __init__(
            self,
            scenario_file:str,
            radius : int = 150
            ):
        self.scenario_file = scenario_file
        self.radius = radius

    def save(self):
        pass

    def process(self):
        df, static_map, scenario_id = load_av2_df(self.scenario_file)
        city = df["city"].values[0]
        agent_id = df["focal_track_id"].values[0]

        local_df = df[df["track_id"] == agent_id].iloc
        origin_position = torch.tensor([local_df[49]])

        timestamps = list(np.sort(df["timestep"].unique()))
        actor_ids = list(df["track_id"].unique())
        num_nodes = len(actor_ids)

        x = torch.zeros(num_nodes, len(timestamps), 2, dtype=torch.float)

        for actor_id, actor_df in df.groupby("track_id"):
