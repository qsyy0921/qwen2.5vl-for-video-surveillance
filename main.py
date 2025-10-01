import json
from agents.scene_agent import SceneAgent
from agents.object_agent import ObjectAgent
from agents.integrate_agent import IntegrateAgent

def run_pipeline(video_path: str):
    scene_agent = SceneAgent()
    object_agent = ObjectAgent()
    integrate_agent = IntegrateAgent()

    print("▶️ [1/3] 场景分析中...")
    scene_result = scene_agent.analyze(video_path)
    print(f"✅ 场景分析结果：\n{scene_result}\n")

    print("▶️ [2/3] 对象分析中...")
    object_result = object_agent.analyze(video_path)
    print(f"✅ 对象分析结果：\n{object_result}\n")

    print("▶️ [3/3] 综合报告生成中...")
    final_report = integrate_agent.analyze(scene_result, object_result)
    print("✅ 综合报告生成完成！\n")

    print("📋 最终综合报告：")
    print(json.dumps(final_report, ensure_ascii=False, indent=2) if isinstance(final_report, dict) else final_report)


if __name__ == "__main__":
    video_path = "/home/l40/newdisk1/mfl/qwenvl/videos/demo.mp4"  # 你的视频路径
    run_pipeline(video_path)
