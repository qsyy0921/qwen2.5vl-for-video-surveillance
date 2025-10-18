# -*- coding: utf-8 -*-
import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def find_frame_column(df):
    # 1) 直给的帧数字段
    for cand in ["video_frames", "frames", "nframes", "frames_used"]:
        if cand in df.columns:
            s = pd.to_numeric(df[cand], errors="coerce")
            if s.notna().any():
                return s.rename("frames_used")

    # 2) 从 video_grid_thw 解析 [[T,H,W]] 的 T
    if "video_grid_thw" in df.columns:
        def parse_t(x):
            if not isinstance(x, str):
                return None
            m = re.search(r"\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*\]", x)
            if m:
                return int(m.group(1))
            return None
        s = df["video_grid_thw"].apply(parse_t)
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().any():
            return s.rename("frames_used")

    # 3) 实在没有，返回全空
    return pd.Series([None] * len(df), name="frames_used")

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)

    # 过滤 EXPLAIN/聚合行，只保留数值行
    numeric_cols = [
        "preprocess_s", "vision_feat_wall_s", "total_wall_s",
        "msg_template_s","read_resize_s","pack_tensor_s","to_device_s",
        "prefill_wall_s","decode_wall_s","generate_wall_s",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 加一列帧数（尽力推断）
    if "frames_used" not in df.columns:
        df["frames_used"] = find_frame_column(df)

    # 只保留这些列里任意有数值的行
    keep_cols = ["frames_used", "preprocess_s", "vision_feat_wall_s", "total_wall_s"]
    exists = [c for c in keep_cols if c in df.columns]
    df = df[exists].copy()
    df = df.dropna(subset=["frames_used"])  # 必须有帧数
    # 阶段时间保留 >=0 的
    for c in ["preprocess_s", "vision_feat_wall_s", "total_wall_s"]:
        if c in df.columns:
            df = df[df[c].isna() | (df[c] >= 0)]

    return df

def scatter_one(ax, x, y, xlabel, ylabel, title):
    ax.scatter(x, y, s=18)  # 不指定颜色（遵守要求）
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="阶段计时 CSV 路径")
    ap.add_argument("--out-dir", default="./plots", help="输出图片目录")
    ap.add_argument("--x-label", default="抽取帧数（T'）")
    ap.add_argument("--title-prefix", default="帧数 vs ")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_and_clean(args.csv)

    if df.empty:
        raise SystemExit("CSV 中没有可用数据（无法解析帧数或阶段时间）。")

    X = df["frames_used"]

    # 要画的三个指标（如果列不存在就跳过）
    targets = [
        ("preprocess_s", "预处理时间（秒）", "preprocess_vs_frames.png"),
        ("vision_feat_wall_s", "视觉特征时间（秒）", "vision_feat_vs_frames.png"),
        ("total_wall_s", "生成总时间（秒）", "generate_total_vs_frames.png"),
    ]

    for col, y_label, fname in targets:
        if col not in df.columns:
            print(f"跳过：CSV 中不存在列 {col}")
            continue
        Y = df[col].dropna()
        # 与 X 对齐
        XY = pd.concat([X, Y], axis=1).dropna()
        if XY.empty:
            print(f"跳过：{col} 没有有效数据")
            continue

        fig = plt.figure(figsize=(6.5, 4.5))
        ax = fig.gca()
        scatter_one(
            ax,
            XY["frames_used"], XY[col],
            args.x_label, y_label,
            f"{args.title_prefix}{y_label}"
        )
        out_path = os.path.join(args.out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"✅ 已保存: {out_path}")

if __name__ == "__main__":
    main()
