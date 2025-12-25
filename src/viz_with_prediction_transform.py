import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from pathlib import Path
from typing import Final, Set, Optional, Tuple

from joblib import Parallel, delayed
from rich.progress import track
import os
# AV2 库
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    _plot_static_map_elements,
    _plot_actor_tracks,
    _OBS_DURATION_TIMESTEPS,
    _PRED_DURATION_TIMESTEPS
)

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

_DEFAULT_N_JOBS: Final[int] = 20

# ==========================================
# 核心逻辑 1: 坐标逆变换 (Local -> Global)
# ==========================================
def transform_local_to_global(
    pred_local: np.ndarray, 
    origin_xy: np.ndarray, 
    origin_heading: float
) -> np.ndarray:
    """
    将局部坐标系的预测轨迹转换回全局坐标系。
    
    Args:
        pred_local: (K, T, 2) 你的预测结果
        origin_xy: (2,) 第50帧时的车辆全局坐标 [x, y]
        origin_heading: float 第50帧时的车辆朝向 (弧度)
        
    Returns:
        pred_global: (K, T, 2) 全局坐标系下的轨迹
    """
    # 构造旋转矩阵 (逆时针旋转)
    c, s = np.cos(origin_heading), np.sin(origin_heading)
    R = np.array([[c, -s], [s, c]]) # (2, 2)

    # 1. 旋转: (K, T, 2) * (2, 2)^T -> (K, T, 2)
    # 使用 einsum 进行批量矩阵乘法
    pred_rotated = np.einsum('kti,ji->ktj', pred_local, R)
    
    # 2. 平移: 加上原点坐标
    pred_global = pred_rotated + origin_xy
    
    return pred_global

def get_focal_agent_state_at_timestep(scenario, timestep: int) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """获取 Focal Agent 在指定时刻的位置和朝向"""
    focal_id = scenario.focal_track_id
    
    # 遍历所有 track 找到 focal track
    for track in scenario.tracks:
        if track.track_id == focal_id:
            # 找到对应时间步的状态
            # object_states 是一个 list，索引对应时间步
            if timestep < len(track.object_states):
                state = track.object_states[timestep]
                if state.observed: # 必须是观测到的
                    return np.array(state.position), state.heading
    return None, None

# ==========================================
# 核心逻辑 2: 自定义可视化 (支持画预测)
# ==========================================
def visualize_scenario_with_prediction(
    scenario, 
    static_map, 
    viz_save_path: Path, 
    prediction_local: Optional[np.ndarray] = None
):
    """
    绘制地图、真值和（转换后的）预测轨迹。
    新增功能：相机自动跟随 Focal Agent，视野更聚焦。
    """
    # === 参数设置 ===
    VIEW_RANGE = 50.0  # [调整这里] 视野半径（米）。数值越小，物体越大（zoom in）。
    # =================
    
    # 1. 准备数据：如果存在预测，先进行坐标转换
    prediction_global = None
    # 归一化时刻（第50帧）
    norm_timestep = _OBS_DURATION_TIMESTEPS - 1 
    
    # 获取归一化原点用于坐标转换
    origin_xy, origin_heading = get_focal_agent_state_at_timestep(scenario, norm_timestep)
    
    if prediction_local is not None:
        if origin_xy is not None:
            prediction_global = transform_local_to_global(prediction_local, origin_xy, origin_heading)
        else:
            print(f"Warning: Cannot find focal agent state at step {norm_timestep} for {scenario.scenario_id}")

    # 2. 开始绘图循环
    frames = []
    total_timesteps = _OBS_DURATION_TIMESTEPS + _PRED_DURATION_TIMESTEPS # 110
    
    # 关闭交互模式
    plt.ioff()

    for timestep in range(total_timesteps):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100) # dpi 调高一点会更清晰
        
        # A. 画地图 (使用官方私有函数)
        _plot_static_map_elements(static_map)
        
        # B. 画真值 (所有物体)
        _plot_actor_tracks(ax, scenario, timestep)
        
        # C. 画预测 (仅在预测阶段显示)
        if prediction_global is not None and timestep >= _OBS_DURATION_TIMESTEPS:
            pred_step_idx = timestep - _OBS_DURATION_TIMESTEPS
            for k in range(prediction_global.shape[0]):
                traj = prediction_global[k] # (60, 2)
                # 虚线轨迹
                ax.plot(traj[:, 0], traj[:, 1], color="#00FF00", alpha=0.4, linewidth=1.5, linestyle="--", zorder=15)
                # 当前预测点
                if pred_step_idx < traj.shape[0]:
                    pos = traj[pred_step_idx]
                    ax.scatter(pos[0], pos[1], color="#00FF00", s=60, edgecolors='black', zorder=20, label="Pred" if k==0 else "")

        # --- [核心修改] 相机跟随逻辑 ---
        # 1. 尝试获取当前帧 Focal Agent 的位置
        cur_pos, _ = get_focal_agent_state_at_timestep(scenario, timestep)
        
        # 2. 如果当前帧没有（比如车还没出现，或者已经消失），则回退使用归一化原点
        if cur_pos is None:
            cur_pos = origin_xy
            
        # 3. 如果找到了位置，就锁定视角
        # if cur_pos is not None:
        #     ax.set_xlim(cur_pos[0] - VIEW_RANGE, cur_pos[0] + VIEW_RANGE)
        #     ax.set_ylim(cur_pos[1] - VIEW_RANGE, cur_pos[1] + VIEW_RANGE)
        # else:
        #     # 实在找不到人（极其罕见），就只设置 aspect equal，不做裁剪
        #     pass
            
        ax.set_aspect("equal")
        plt.axis("off")
        plt.tight_layout()
        # ---------------------------
        
        # D. 保存帧 (修复 tostring_rgb 报错的版本)
        fig.canvas.draw()
        img_rgba = np.asarray(fig.canvas.buffer_rgba())
        img_rgb = img_rgba[:, :, :3]
        
        frames.append(img_rgb)
        plt.close(fig)

    # 3. 合成视频
    if frames:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(viz_save_path), fourcc, 10, (width, height))
        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()

def quick_get_parquet_files(data_dir):
    parquet_files = []
    # os.walk 结合 os.scandir 性能更佳
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    return parquet_files

# ==========================================
# 主流程
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="生成带预测轨迹的可视化视频")
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/home/ubuntu/DISK2/ZJT/argoverse_dataset_v2/train",
        help="Argoverse 场景数据（.parquet）存放的根目录。"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/ubuntu/DISK2/ZJT/sept/src/datamiming/good_vedio",
        help="生成的视频保存目录。"
    )
    parser.add_argument(
        "--ids_file", 
        type=str, 
        default="/home/ubuntu/DISK2/ZJT/sept/src/datamiming/good_case.txt",
        help="包含 scenario_id 的 txt 文件路径。"
    )
    parser.add_argument("--pred_file", 
                        type=str, 
                        default="/home/ubuntu/DISK2/ZJT/sept/src/datamiming/1224_val_predictions_dict.pkl",
                        help="预测结果 pkl 文件路径")

    parser.add_argument(
        "--n_jobs", 
        type=int, 
        default=_DEFAULT_N_JOBS, 
        help="并行处理的进程数。"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="启用调试模式，单线程运行。"
    )

    
    args = parser.parse_args()
    
    # 加载数据
    with open(args.ids_file, 'r') as f:
        target_ids = {line.strip() for line in f if line.strip()}
    
    print(f"Loading predictions from {args.pred_file}...")
    with open(args.pred_file, 'rb') as f:
        all_preds = pickle.load(f) # 期望是 {id: numpy_array}
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning data dir using os.walk...")
    selected_files = []
    
    # 直接在扫描过程中进行匹配，效率最高且不会报错
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith(".parquet"):
                # 从文件名提取 ID (假设格式为 scenario_ID.parquet 或 ID.parquet)
                # file 是字符串，先用 Path(file).stem 获取不带后缀的文件名
                file_stem = Path(file).stem
                scenario_id = file_stem.split("_")[-1]
                
                if scenario_id in target_ids:
                    full_path = Path(root) / file
                    selected_files.append(full_path)

    print(f"Found {len(selected_files)} / {len(target_ids)} matched scenarios.")
    

    def process_one(path):
        sid = path.stem.split("_")[-1]
        map_path = path.parents[0] / f"log_map_archive_{sid}.json"
        save_path = output_dir / f"{sid}_viz.mp4"
        
        # 获取该场景的预测
        pred = all_preds.get(sid) # 这里拿到的是 local coordinates 的预测
        
        try:
            scenario = scenario_serialization.load_argoverse_scenario_parquet(path)
            static_map = ArgoverseStaticMap.from_json(map_path)
            
            # 调用自定义绘图函数
            visualize_scenario_with_prediction(scenario, static_map, save_path, pred)
            
        except Exception as e:
            print(f"Error {sid}: {e}")

    if args.debug:
        for p in track(selected_files):
            process_one(p)
    else:
        Parallel(n_jobs=args.n_jobs)(
            delayed(process_one)(p) for p in track(selected_files)
        )

if __name__ == "__main__":
    main()