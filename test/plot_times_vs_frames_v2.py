# -*- coding: utf-8 -*-
"""
视频处理时间分析工具 - 重写版
功能：绘制三个运行阶段（预处理/特征提取/生成）的时间与视频帧数的关系散点图
特点：
- 支持多轮运行(run1/2/3)数据对比
- 自动数据清洗和类型转换
- 可视化优化：颜色区分、标记形状、jitter防重叠
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import io

# 内嵌测试数据
CSV_DATA = """run,idx,total,video,frames,pre_s,feat_s,gen_s
1,1,50,Abuse001_x264.mp4,2729,0.597,3.121,6.653
1,2,50,Abuse002_x264.mp4,865,0.311,0.874,3.959
... (完整数据保持不变) ...
3,50,50,Abuse050_x264.mp4,4862,0.782,2.503,13.477
"""

class VideoTimeAnalyzer:
    def __init__(self):
        self.df = None
        # 可视化配置
        self.colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}  # 蓝/橙/绿
        self.markers = {1: "o", 2: "s", 3: "^"}  # 圆/方/三角
        self.zorders = {1: 3, 2: 2, 3: 1}  # 图层顺序

    def _clean_data(self, df):
        """数据清洗和类型转换"""
        # 去除字符串前后空格
        df = df.applymap(lambda s: s.strip() if isinstance(s, str) else s)
        
        # 数值列转换
        for col in ["run", "idx", "total", "frames"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        # 提取时间列中的第一个浮点数
        for col in ["pre_s", "feat_s", "gen_s"]:
            df[col] = df[col].apply(lambda x: float(re.search(r"[-+]?\d*\.?\d+", str(x)).group(0)) 
                                   if pd.notna(x) and re.search(r"[-+]?\d*\.?\d+", str(x)) 
                                   else np.nan)
            
        # 移除无效行并转换类型
        df = df.dropna(subset=["run", "frames", "pre_s", "feat_s", "gen_s"])
        df["run"] = df["run"].astype(int)
        df["frames"] = df["frames"].astype(int)
        
        return df

    def load_data(self):
        """加载并清洗数据"""
        self.df = pd.read_csv(io.StringIO(CSV_DATA), dtype=str)
        self.df = self._clean_data(self.df)
        return self.df

    def calculate_jitter(self):
        """计算合适的jitter值"""
        med_frames = np.median(self.df["frames"])
        return int(max(8, min(0.005 * med_frames, 60)))

    def plot_stage(self, y_col, title, output_path):
        """绘制单个阶段的散点图"""
        plt.figure(figsize=(10, 6), dpi=120)
        plt.grid(True, linestyle="--", alpha=0.3)
        
        jitter = self.calculate_jitter()
        runs = sorted(self.df["run"].unique())
        offsets = np.linspace(-jitter, jitter, num=len(runs))
        
        # 按zorder从小到大绘制，确保run1在最上层
        for i, run in enumerate(sorted(runs, key=lambda r: (r != 1, r))):
            subset = self.df[self.df["run"] == run]
            plt.scatter(
                x=subset["frames"] + offsets[i],
                y=subset[y_col],
                s=55 if run == 1 else 50,  # run1点稍大
                c=self.colors[run],
                marker=self.markers[run],
                alpha=0.85,
                edgecolors="#333",
                linewidths=1.2 if run == 1 else 0.8,
                label=f"Run {run}",
                zorder=self.zorders[run]
            )

        plt.xlabel("Video Frames Count", fontsize=11)
        plt.ylabel(f"{title} Time (seconds)", fontsize=11)
        plt.title(f"{title} Time vs Video Frames", pad=12, fontsize=12)
        
        # 创建自定义图例
        legend_elements = [
            Line2D([0], [0], marker=self.markers[run], color='w',
                   markerfacecolor=self.colors[run], markersize=10,
                   markeredgecolor='#333', label=f'Run {run}')
            for run in runs
        ]
        
        plt.legend(handles=legend_elements, title="Run",
                  loc="center left", bbox_to_anchor=(1, 0.5),
                  frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Chart saved: {output_path}")

    def generate_all_charts(self):
        """生成所有分析图表"""
        output_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_stage("pre_s", "Preprocessing", 
                       os.path.join(output_dir, "pre_vs_frames.png"))
        self.plot_stage("feat_s", "Feature Extraction", 
                       os.path.join(output_dir, "feat_vs_frames.png"))
        self.plot_stage("gen_s", "Generation", 
                       os.path.join(output_dir, "gen_vs_frames.png"))

if __name__ == "__main__":
    analyzer = VideoTimeAnalyzer()
    df = analyzer.load_data()
    
    print("Detected runs:", sorted(df["run"].unique()))
    print("Data count per run:\n", df.groupby("run")["frames"].count())
    
    analyzer.generate_all_charts()
