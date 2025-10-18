# main.py
import json
import time
from agents.scene_agent import SceneAgent
from agents.object_agent import ObjectAgent
from agents.integrate_agent import IntegrateAgent

def run_pipeline(video_path: str):
    # 创建 agent 实例
    scene_agent = SceneAgent()
    object_agent = ObjectAgent()
    integrate_agent = IntegrateAgent()

    # 记录开始时间
    start_time = time.time()

    # 场景分析
    scene_start_time = time.time()  # 记录场景分析开始时间
    scene_result = scene_agent.analyze(video_path)
    scene_end_time = time.time()  # 记录场景分析结束时间
    scene_analysis_time = scene_end_time - scene_start_time  # 计算场景分析时间
    print("场景分析完成。\n")

    # 对象分析
    object_start_time = time.time()  # 记录对象分析开始时间
    object_result = object_agent.analyze(video_path)
    object_end_time = time.time()  # 记录对象分析结束时间
    object_analysis_time = object_end_time - object_start_time  # 计算对象分析时间
    print("对象分析完成。\n")

    # 综合报告生成
    integrate_start_time = time.time()  # 记录综合报告生成开始时间
    final_report = integrate_agent.analyze(scene_result, object_result)
    integrate_end_time = time.time()  # 记录综合报告生成结束时间
    integrate_analysis_time = integrate_end_time - integrate_start_time  # 计算综合报告时间
    print("综合报告生成完成！\n")

    # 打印最终报告
    print("📋 最终综合报告：")
    print(json.dumps(final_report, ensure_ascii=False, indent=2) if isinstance(final_report, dict) else final_report)

    # 计算总推理时间
    end_time = time.time()
    total_time = end_time - start_time  # 总时间（秒）

    # 输出每个步骤的时间和总时间
    print(f"\n⏱️ 场景分析时间：{scene_analysis_time:.2f} 秒")
    print(f"⏱️ 对象分析时间：{object_analysis_time:.2f} 秒")
    print(f"⏱️ 综合报告生成时间：{integrate_analysis_time:.2f} 秒")
    print(f"⏱️ 总推理时间：{total_time:.2f} 秒")

if __name__ == "__main__":
    video_path = "/home/l40/newdisk1/mfl/videosur/data/videos/car.mp4"  # 你的视频路径
    run_pipeline(video_path)
