# 完成论文中的数据提取，以及数据预处理
from pathlib import Path
from sympy import rad, true
from ..av2_data_utils import load_av2_df, OBJECT_TYPE_MAP, OBJECT_TYPE_MAP_COMBINED, LaneTypeMap
import torch
import numpy as np
from av2.map.map_api import ArgoverseStaticMap
import av2.geometry.interpolate as interp_utils


class Av2Extractor:
    def __init__(
            self,
            save_path: Path,
            radius: int = 150,
            remove_outlier_actors: bool = true
    ):
        self.save_path = save_path
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors

    def save(self, file: Path):
        scenario_id = file.stem.split("_")[-1]
        output_path = self.save_path/f"{scenario_id}.pt"
        self.save_path.mkdir(parents=True, exist_ok=True)
        data = self.process(scenario_file=file)
        torch.save(data, output_path)

    def process(self, scenario_file: Path):
        ###
        # 1.提取焦点目标，根据第49time归一化坐标
        # 2.提取150m之内的目标
        # 3.返回position离散矢量、heading、velocity、attr等属性
        ###
        df, am, scenario_id = load_av2_df(scenario_file)
        # 提取焦点目标序列
        agent_id = df["focal_track_id"].values[0]
        local_df = df[df["track_id"] == agent_id].iloc

        origin_position = torch.tensor(
            [local_df[49]["position_x"], local_df[49]["position_y"]], dtype=torch.float)
        origin_theta = torch.tensor(local_df[49]["heading"], dtype=torch.float)
        rotate_theta = torch.tensor(
            [
                [torch.cos(origin_theta), -torch.sin(origin_theta)],
                [torch.sin(origin_theta), torch.cos(origin_theta)],
            ]
        )
        city = df['city'].values[0]

        # 提取150m内的单位
        timestamps = list(np.sort(df["timestep"].unique()))
        cur_df = df[df["timestep"] == timestamps[49]]
        cur_pos = torch.from_numpy(
            cur_df[["position_x", "position_y"]].values).float()
        out_of_range = np.linalg.norm(
            cur_pos - origin_position, axis=1) > self.radius
        actor_ids = list(cur_df["track_id"].unique())
        actor_ids = [aid for i, aid in enumerate(
            actor_ids) if not out_of_range[i]]
        actor_ids.remove(agent_id)
        actor_ids = [agent_id] + actor_ids  # move the focal agent to the first
        num_nodes = len(actor_ids)

        df = df[df["track_id"].isin(actor_ids)]  # finish the filter

        x_position = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_src = torch.zeros(num_nodes, 50, 8, dtype=torch.float)
        x_diff = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_heading = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_velocity = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_attr = torch.zeros(num_nodes, 3, dtype=torch.float)
        padding_mask = torch.ones(num_nodes, 110, dtype=torch.bool)
        # 按每个id目标提取对应信息
        for actor_id, actor_df in df.groupby("track_id"):
            node_idx = actor_ids.index(actor_id)
            object_type = OBJECT_TYPE_MAP[actor_df["object_type"].values[0]]

            x_attr[node_idx, 0] = object_type  # 物体类别
            # object categories
            x_attr[node_idx, 1] = actor_df["object_category"].values[0]
            # 在地图上的交通形式
            x_attr[node_idx, 2] = OBJECT_TYPE_MAP_COMBINED[actor_df["object_type"].values[0]]

            node_steps = [timestamps.index(ts) for ts in actor_df["timestep"]]

            # padding筛选出可见时间
            padding_mask[node_idx, node_steps] = False

            pos_xy = torch.from_numpy(
                np.stack(
                    [actor_df["position_x"].values, actor_df["position_y"].values],
                    axis=-1
                )
            ).float()
            heading = torch.from_numpy(actor_df["heading"].values).float()
            velocity = torch.from_numpy(
                actor_df[["velocity_x", "velocity_y"]].values).float()
            velocity_norm = torch.norm(velocity, dim=1)
            x_position[node_idx, node_steps, :2] = torch.matmul(
                pos_xy - origin_position, rotate_theta)
            x_heading[node_idx, node_steps] = (
                heading - origin_theta + np.pi) % (2 * np.pi) - np.pi
            x_velocity[node_idx, node_steps] = velocity_norm

        (
            lane_positions,
            is_intersections,
            lane_ctrs,
            lane_angles,
            lane_attr,
            lanes_start,
            lanes_end,
            lanes_length,
            lane_padding_mask,
        ) = self.get_lane_features(am, origin_position, origin_position, rotate_theta, self.radius)

        if self.remove_outlier_actors:
            lane_samples = lane_positions.view(-1, 2)
            nearest_dist = torch.cdist(
                x_position[:, 49, :2], lane_samples).min(dim=1).values
            valid_actor_mask = nearest_dist < 5
            valid_actor_mask[0] = True  # always keep the target agent

            x_position = x_position[valid_actor_mask]
            x_diff = x_diff[valid_actor_mask]
            x_heading = x_heading[valid_actor_mask]
            x_velocity = x_velocity[valid_actor_mask]
            x_attr = x_attr[valid_actor_mask]
            padding_mask = padding_mask[valid_actor_mask]
            num_nodes = x_position.shape[0]

        x_center = x_position[:, 49].clone()
        x_diff[:, 1:50] = torch.where((padding_mask[:, 1:50] | padding_mask[:, :49]).unsqueeze(
            -1), torch.zeros(num_nodes, 49, 2), x_position[:, 1:50] - x_position[:, :49])
        x_diff[:, 50:] = torch.where((padding_mask[:, 50:] | padding_mask[:, 49].unsqueeze(1)).unsqueeze(-1),
                                     torch.zeros(num_nodes, 60, 2),
                                     x_position[:, 50:] -
                                     x_position[:, 49].unsqueeze(1)
                                     )
        x_diff[:, 0] = torch.zeros(num_nodes, 2)
        y_diff = x_diff[:, 50:]

        x_velocity_diff = x_velocity[:, :50].clone()
        x_velocity_diff[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]),
            torch.zeros(num_nodes, 49),
            x_velocity_diff[:, 1:50] - x_velocity_diff[:, :49],
        )
        x_velocity_diff[:, 0] = torch.zeros(num_nodes)

        A, T, _ = x_diff[:, :50].shape

        time_stamp = torch.arange(0, T, device=x_diff.device).float()
        time_stamp = time_stamp.view(1, T, 1).expand(A, -1, -1)

        x_src = torch.concat([x_diff[:, :50],  # A 50 2
                              x_velocity_diff.unsqueeze(-1),
                              (~padding_mask[:, :50].unsqueeze(-1)).float(),
                              time_stamp],
                             dim=-1)

        lane_normalized = lane_positions - lane_ctrs.unsqueeze(-2)
        lane_src = torch.concat([lane_normalized,
                                 (~lane_padding_mask.unsqueeze(-1)).float()],  # L D
                                dim=-1)

        agent_heading = x_heading[:, 49]  # [nums 1]
        agent_angles = torch.stack(
            [torch.cos(agent_heading), torch.sin(agent_heading)], dim=-1)  # [nums 2]

        road_angles = torch.stack(
            [torch.cos(lane_angles), torch.sin(lane_angles)], dim=-1)  # [nums 2]

        agent_pos_feat = torch.cat([x_center, agent_angles], dim=-1)
        road_pos_feat = torch.cat([lane_ctrs, road_angles], dim=-1)

        return {
            "x_src": x_src,
            "lane_src": lane_src,
            "agent_pos_feat": agent_pos_feat,
            "road_pos_feat": road_pos_feat,
            "y_diff": y_diff,

            "x_padding_mask": padding_mask,

            # "x_diff": x_diff[:, :50],
            # "x_position": x_position,
            # "x_heading": x_heading,
            # "x_velocity": x_velocity,
            "x_attr": x_attr,
            # "x_center": x_center,

            # "lane_positions": lane_positions,
            # "lane_centers": lane_ctrs,
            # "lane_angles": lane_angles,
            "lane_attr": lane_attr,
            # "lanes_start": lanes_start,
            # "lanes_end": lanes_end,
            # "lanes_length": lanes_length,
            "lane_padding_mask": lane_padding_mask,
            # "is_intersections": is_intersections,

            # "scenario_id": scenario_id,
            # "track_id": agent_id,
            # "city": city,
        }

    @staticmethod
    def get_lane_features(
        am: ArgoverseStaticMap,
        query_pos: torch.Tensor,
        origin: torch.Tensor,
        rotate_mat: torch.Tensor,
        radius: float,
    ):
        lane_segments = am.get_nearby_lane_segments(query_pos.numpy(), radius)

        lane_positions, is_intersections, lane_attrs = [], [], []
        for segment in lane_segments:
            lane_centerline, lane_width = interp_utils.compute_midpoint_line(
                left_ln_boundary=segment.left_lane_boundary.xyz,
                right_ln_boundary=segment.right_lane_boundary.xyz,
                num_interp_pts=20,
            )
            lane_centerline = torch.from_numpy(
                lane_centerline[:, :2]).float()
            lane_centerline = torch.matmul(
                lane_centerline - origin, rotate_mat)
            is_intersection = am.lane_is_in_intersection(segment.id)

            lane_positions.append(lane_centerline)
            is_intersections.append(is_intersection)

            # get lane attrs
            lane_type = LaneTypeMap[segment.lane_type]
            attribute = torch.tensor(
                [lane_type, lane_width, is_intersection], dtype=torch.float
            )
            lane_attrs.append(attribute)

        lane_positions = torch.stack(lane_positions)
        lanes_start = lane_positions[:, 0, :]
        lanes_end = lane_positions[:, -1, :]
        lanes_length = torch.norm(lanes_end-lanes_start, dim=1)
        lanes_ctr = lane_positions[:, 9:11].mean(dim=1)
        lanes_angle = torch.atan2(
            lane_positions[:, 10, 1] - lane_positions[:, 9, 1],
            lane_positions[:, 10, 0] - lane_positions[:, 9, 0],
        )
        is_intersections = torch.Tensor(is_intersections)
        lane_attrs = torch.stack(lane_attrs, dim=0)

        x_max, x_min = radius, -radius
        y_max, y_min = radius, -radius

        padding_mask = (
            (lane_positions[:, :, 0] > x_max)
            | (lane_positions[:, :, 0] < x_min)
            | (lane_positions[:, :, 1] > y_max)
            | (lane_positions[:, :, 1] < y_min)
        )

        invalid_mask = padding_mask.all(dim=-1)
        lane_positions = lane_positions[~invalid_mask]
        is_intersections = is_intersections[~invalid_mask]
        lane_attrs = lane_attrs[~invalid_mask]
        lanes_ctr = lanes_ctr[~invalid_mask]
        lanes_angle = lanes_angle[~invalid_mask]
        padding_mask = padding_mask[~invalid_mask]
        lanes_start = lanes_start[~invalid_mask]
        lanes_end = lanes_end[~invalid_mask]
        lanes_length = lanes_length[~invalid_mask]

        lane_positions = torch.where(
            padding_mask[..., None], torch.zeros_like(
                lane_positions), lane_positions
        )

        return (
            lane_positions,
            is_intersections,
            lanes_ctr,
            lanes_angle,
            lane_attrs,
            lanes_start,
            lanes_end,
            lanes_length,
            padding_mask,
        )
