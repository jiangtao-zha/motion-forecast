from datamodule.Post.av2_extractor import Av2Extractor
from pathlib import Path
from datamodule.av2_data_utils import load_av2_df

path = Path("/home/ubuntu/DISK2/ZJT/argoverse_dataset_v2/train")
file = "b91de4c0-c398-460e-9bc0-85e10ab3ba16/scenario_b91de4c0-c398-460e-9bc0-85e10ab3ba16.parquet"

imput_file = path / file

extractor = Av2Extractor(Path("/home/ubuntu/DISK2/ZJT/sept/src"))

df, static_map, scenario_id = load_av2_df(scenario_file=imput_file)
df.to_csv(f"{scenario_id}.csv")

extractor.save(imput_file)