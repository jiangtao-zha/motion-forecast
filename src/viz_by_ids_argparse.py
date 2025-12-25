import argparse
from pathlib import Path
from typing import Final, Set

from joblib import Parallel, delayed
from rich.progress import track

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    visualize_scenario,
)
from av2.map.map_api import ArgoverseStaticMap

# 默认并行进程数：使用除一个 CPU 之外的所有 CPU
_DEFAULT_N_JOBS: Final[int] = -2 

def load_target_ids(txt_path: Path) -> Set[str]:
    """从文本文件中读取场景 ID。"""
    if not txt_path.exists():
        raise FileNotFoundError(f"未找到 ID 文件: {txt_path}")
    
    with open(txt_path, 'r') as f:
        # 读取每一行并去除多余空格
        return {line.strip() for line in f if line.strip()}

def generate_visualization(scenario_path: Path, viz_output_dir: Path) -> None:
    """为单个场景生成并保存动态可视化视频。"""
    # 提取 ID：假设文件名格式符合 Argoverse 标准
    scenario_id = scenario_path.stem.split("_")[-1]
    
    # 静态地图通常存放在与场景文件相同的目录中
    static_map_path = (
        scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
    )
    viz_save_path = viz_output_dir / f"{scenario_id}.mp4"

    try:
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        visualize_scenario(scenario, static_map, viz_save_path)
    except Exception as e:
        print(f"处理场景 {scenario_id} 时出错: {e}")

def main():
    # 使用 argparse 配置命令行输入
    parser = argparse.ArgumentParser(description="根据 txt 文件中的 ID 列表生成 Argoverse 场景可视化。")
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/home/ubuntu/DISK2/ZJT/argoverse_dataset_v2/train",
        help="Argoverse 场景数据（.parquet）存放的根目录。"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/ubuntu/DISK2/ZJT/sept/src/datamiming/bad_vedio",
        help="生成的视频保存目录。"
    )
    parser.add_argument(
        "--ids_file", 
        type=str, 
        default="/home/ubuntu/DISK2/ZJT/sept/src/datamiming/bad_case.txt",
        help="包含 scenario_id 的 txt 文件路径。"
    )
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

    # 路径处理
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ids_file = Path(args.ids_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载目标 ID
    target_ids = load_target_ids(ids_file)
    print(f"已从文件中加载 {len(target_ids)} 个 ID。")

    # 2. 扫描数据集目录
    print(f"正在扫描目录: {data_dir} ...")
    all_files = list(data_dir.rglob("*.parquet"))
    
    # 3. 匹配文件
    selected_files = [
        p for p in all_files if p.stem.split("_")[-1] in target_ids
    ]
    
    print(f"匹配成功: {len(selected_files)} / {len(target_ids)}")

    if not selected_files:
        print("未找到匹配的场景文件，请检查目录或 ID 是否正确。")
        return

    # 4. 执行渲染
    if args.debug:
        # 单线程模式
        for scenario_path in track(selected_files, description="Rendering"):
            generate_visualization(scenario_path, output_dir)
    else:
        # 并行模式
        Parallel(n_jobs=args.n_jobs)(
            delayed(generate_visualization)(scenario_path, output_dir)
            for scenario_path in track(selected_files, description="Rendering")
        )

if __name__ == "__main__":
    main()