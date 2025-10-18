# plot_times_vs_frames.py
import argparse, os, re
import pandas as pd
import matplotlib.pyplot as plt

RUNS = [1, 2, 3]
COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}  # 蓝/橙/绿
MARKERS = {1: "o", 2: "s", 3: "^"}                   # 圆/方/三角
LABELS_ZH = {"pre":"预处理时间 (s)", "feat":"特征提取时间 (s)", "gen":"生成时间 (s)"}

def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns: 
            return c
    # 不区分大小写再试
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low: 
            return low[c.lower()]
    return None

def detect_frames_col(df):
    return pick_column(df, ["frames","frame","num_frames","nframes","video_frames","total_frames"])

def detect_long_format(df):
    # 有 run 列且有 pre/feat/gen（或 *_s）列就当作“长表”
    if pick_column(df, ["run","Run","RUN"]):
        for s in ["pre","feat","gen"]:
            if pick_column(df, [s, f"{s}_s", s.upper(), f"{s}_S"]):
                return True
    return False

def detect_wide_mapping(df, stage):
    """ 返回 {run: colname}，适配 pre_run1 / pre_r1 / pre1 / pre_s_run2 等写法 """
    mapping = {}
    for col in df.columns:
        lc = col.lower()
        if not lc.startswith(stage): 
            continue
        # 尝试多种后缀：_run1 / run1 / _r1 / r1 / 末尾数字
        m = (re.search(r'run[_\- ]?(\d+)', lc) or
             re.search(r'[_\-]r(\d+)', lc) or
             re.search(r'(\d+)$', lc))
        if m:
            r = int(m.group(1))
            if r in RUNS:
                mapping[r] = col
    return mapping

def get_long_col(df, base):
    return pick_column(df, [base, f"{base}_s", base.upper(), f"{base}_S"])

def plot_stage(df, frames_col, stage, out_path, is_long):
    plt.figure(figsize=(8,6))
    if is_long:
        run_col = pick_column(df, ["run","Run","RUN"])
        y_col = get_long_col(df, stage)
        if y_col is None:
            raise ValueError(f"未找到列：{stage} / {stage}_s（长表）")
        printed = []
        for r in RUNS:
            sub = df[df[run_col] == r]
            if sub.empty or sub[frames_col].isna().all() or sub[y_col].isna().all():
                continue
            plt.scatter(sub[frames_col], sub[y_col], s=64,
                        c=COLORS[r], marker=MARKERS[r], edgecolors="k", linewidths=0.8,
                        alpha=0.9, label=f"Run {r}")
            printed.append((r, len(sub)))
        print(f"[{stage}] 长表：使用列 frames={frames_col}, y={y_col}，各 run 点数：{printed}")
    else:
        # 宽表：pre_run1/pre_r1/pre1...
        mapping = detect_wide_mapping(df, stage)
        if not mapping:
            raise ValueError(f"未在表头找到 {stage} 对应的 run 列（如 {stage}_run1 / {stage}_r1 / {stage}1）")
        printed = []
        for r in RUNS:
            col = mapping.get(r)
            if col is None: 
                continue
            sub = df[[frames_col, col]].dropna()
            if sub.empty: 
                continue
            plt.scatter(sub[frames_col], sub[col], s=64,
                        c=COLORS[r], marker=MARKERS[r], edgecolors="k", linewidths=0.8,
                        alpha=0.9, label=f"Run {r}")
            printed.append((r, len(sub)))
        print(f"[{stage}] 宽表：使用列 frames={frames_col}, y_map={mapping}，各 run 点数：{printed}")

    plt.xlabel("视频帧数", fontsize=12)
    plt.ylabel(LABELS_ZH[stage], fontsize=12)
    plt.title(f"{LABELS_ZH[stage]} vs 帧数", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="运行次数", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV 文件路径")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print("读取到列：", list(df.columns))

    frames_col = detect_frames_col(df)
    if frames_col is None:
        raise ValueError("未找到帧数字段（候选：frames/frame/num_frames/nframes/video_frames/total_frames）")

    is_long = detect_long_format(df)
    print("判定表格格式：", "长表（有 run 列）" if is_long else "宽表（每个阶段有 run1/run2/run3 列）")

    out_dir = os.path.dirname(args.csv) or "."
    plot_stage(df, frames_col, "pre",  os.path.join(out_dir, "pre_vs_frames.png"),  is_long)
    plot_stage(df, frames_col, "feat", os.path.join(out_dir, "feat_vs_frames.png"), is_long)
    plot_stage(df, frames_col, "gen",  os.path.join(out_dir, "gen_vs_frames.png"),  is_long)
    print("✅ 已输出：pre_vs_frames.png, feat_vs_frames.png, gen_vs_frames.png")

if __name__ == "__main__":
    main()
