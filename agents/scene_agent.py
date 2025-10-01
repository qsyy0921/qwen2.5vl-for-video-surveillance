# import requests
# import json

# API_URL = "http://127.0.0.1:8000/infer"

# class SceneAgent:
#     """
#     场景智能体：负责全局语义场景理解
#     """
#     def analyze(self, video_path: str):
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "video",
#                         "video": f"file://{video_path}",
#                         "max_pixels": 151200,
#                         "fps": 1.0
#                     },
#                     {
#                         "type": "text",
#                         "text": (
#                             "请分析这段监控视频的整体场景，描述主要事件、环境背景、"
#                             "发生的时间/地点特征，以及你对整个事件的理解。"
#                         )
#                     }
#                 ]
#             }
#         ]

#         resp = requests.post(API_URL, json=messages)
#         data = resp.json()
#         return data.get("result", [""])[0] if isinstance(data.get("result"), list) else data



# agents/scene_agent.py
import requests

API_URL = "http://127.0.0.1:8000/infer"

class SceneAgent:
    """负责全局场景分析"""
    def __init__(self, session=None):
        self.s = session or requests.Session()

    def analyze(self, video_path: str):
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{video_path}", "fps": 1.0},
                {"type": "text", "text": "请分析该监控视频的整体场景、主要事件和环境背景，尽量简洁精炼。"}
            ]
        }]
        r = self.s.post(API_URL, json=messages, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data.get("result", data)
