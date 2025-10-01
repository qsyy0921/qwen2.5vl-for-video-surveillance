import requests

API_URL = "http://127.0.0.1:8000/infer"

class ObjectAgent:
    """
    对象智能体：分析视频中的主要对象、状态和交互关系
    """
    def analyze(self, video_path: str):
        # 生成请求体，完全符合 server.py 的接口要求
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
                            "请仔细分析这段视频中的主要对象："
                            "列出出现的所有关键对象（例如人、车、包裹、设备等），"
                            "描述每个对象的数量、外观特征、动作状态（静止/移动/交互），"
                            "并指出对象之间的关系（例如搬运、交谈、冲突、合作）。"
                        )
                    }
                ]
            }
        ]

        try:
            resp = requests.post(API_URL, json=messages)
            resp.raise_for_status()  # 捕获 HTTP 错误
            data = resp.json()
            if isinstance(data, dict) and "result" in data:
                result = data["result"]
                return result[0] if isinstance(result, list) else result
            else:
                return data
        except Exception as e:
            return {"error": str(e)}
