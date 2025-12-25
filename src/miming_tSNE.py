import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import os

def run_tsne_analysis(pt_path, csv_path, output_img="tsne_result.png", sample_n=5000):
    print(">>> 1. Loading data...")
    # 加载特征
    feats = torch.load(pt_path, map_location='cpu',weights_only=True).numpy()
    # 加载标签 (Scenario ID, FDE, Entropy, is_miss)
    df = pd.read_csv(csv_path)

    # --- 关键检查：行数必须对齐 ---
    if len(feats) != len(df):
        print(f"\033[91mError: Size mismatch! Feats: {len(feats)}, CSV: {len(df)}\033[0m")
        print("Cannot proceed because we don't know which ID belongs to which Feature.")
        return

    print(f"Data loaded. Total samples: {len(df)}")
    print(f"Feature dimension: {feats.shape[1]}")

    # --- 2. 抽样 (为了速度) ---
    # 如果数据量太大，t-SNE 会跑很久。先随机抽样一部分来看分布。
    if len(df) > sample_n:
        print(f"Dataset too large. Sampling {sample_n} random samples for t-SNE...")
        indices = np.random.choice(len(df), sample_n, replace=False)
        feats_subset = feats[indices]
        df_subset = df.iloc[indices].copy()
    else:
        feats_subset = feats
        df_subset = df.copy()

    print(">>> 2. Running t-SNE (this may take a while)...")
    # perplexity: 困惑度，通常 30-50。
    # init='pca': 通常能让结果更稳定
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(feats_subset)

    # 把结果存回 DataFrame
    df_subset['tsne_x'] = tsne_results[:, 0]
    df_subset['tsne_y'] = tsne_results[:, 1]

    print(">>> 3. Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 左图：按 FDE 误差上色 (连续值) ---
    # 颜色越深/越红，误差越大
    sns.scatterplot(
        ax=axes[0],
        data=df_subset,
        x='tsne_x', y='tsne_y',
        hue='top1_fde', # 你的 CSV 里叫 fde 还是 min_fde，请确认列名
        palette='viridis',
        size='top1_fde', sizes=(1, 100), # 误差大的点画大一点
        alpha=0.7
    )
    axes[0].set_title(f't-SNE colored by FDE (Sample N={len(df_subset)})')

    # --- 右图：按 Miss 状态上色 (离散值) ---
    # 假设 CSV 里有 'is_miss' 列，如果没有，我们现场造一个 (FDE > 2.0)
    if 'is_miss' not in df_subset.columns:
        df_subset['is_miss'] = df_subset['fde'] > 2.0
    
    # 定义颜色：Miss (True) 为红色，正常 (False) 为灰色
    custom_palette = {True: 'red', False: 'lightgray'}
    
    sns.scatterplot(
        ax=axes[1],
        data=df_subset,
        x='tsne_x', y='tsne_y',
        hue='is_miss',
        palette=custom_palette,
        style='is_miss', # Miss 用 X 表示，正常用 圆点
        markers={True: 'X', False: 'o'},
        alpha=0.6
    )
    axes[1].set_title('t-SNE Highlight Miss Cases (Red)')

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Done! Plot saved to {output_img}")
    
    # (可选) 打印几个落在“孤岛”簇里的难例 ID，供你后续去可视化
    # 简单的逻辑：找出 Miss 且 t-SNE 坐标比较偏的点
    miss_samples = df_subset[df_subset['is_miss'] == True]
    if not miss_samples.empty:
        print("\n>>> Top 5 Hard Cases IDs (High FDE) from this sample:")
        print(miss_samples.sort_values(by='top1_fde', ascending=False)['id'].head(5).values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', type=str,  default="/home/ubuntu/DISK2/ZJT/sept/src/datamiming/1224_val_latent_features.pt" ,help='Path to .pt features')
    parser.add_argument('--csv', type=str, default="/home/ubuntu/DISK2/ZJT/sept/src/datamiming/mining_results.csv",help='Path to .csv metrics')
    parser.add_argument('--out', type=str, default='/home/ubuntu/DISK2/ZJT/sept/src/datamiming/1224_tsne_viz.png', help='Output image path')
    parser.add_argument('--limit', type=int, default=8000, help='Number of points to sample')
    args = parser.parse_args()

    run_tsne_analysis(args.pt, args.csv, args.out, args.limit)