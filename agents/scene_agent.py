import requests
import json

API_URL = "http://127.0.0.1:8000/infer"

class SceneAgent:
    """
    场景智能体：负责全局语义场景理解
    """
    def analyze(self, video_path: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": 151200,
                        "fps": 1.0
                    },
                    {
                        "type": "text",
                        "text": (
                            "请分析这段监控视频的整体场景，描述主要事件、环境背景、"
                            "发生的时间/地点特征，以及你对整个事件的理解。"
                        )
                    }
                ]
            }
        ]

        resp = requests.post(API_URL, json=messages)
        data = resp.json()
        return data.get("result", [""])[0] if isinstance(data.get("result"), list) else data
