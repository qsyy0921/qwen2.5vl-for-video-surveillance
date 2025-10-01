# 

# agents/integrate_agent.py
import requests

API_URL = "http://127.0.0.1:8000/infer"

class IntegrateAgent:
    """融合场景与对象分析，输出综合结论"""
    def __init__(self, session=None):
        self.s = session or requests.Session()

    def analyze(self, scene_result, object_result):
        prompt = (
            f"将以下两段分析结果融合为一段综合描述，简要总结事件核心内容：\n\n"
            f"【场景分析】{scene_result}\n\n"
            f"【对象分析】{object_result}\n\n"
            f"请输出一个结构化 JSON，包含 'summary'（一句话摘要）和 'suggestions'（三条处置建议）。"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        r = self.s.post(API_URL, json=messages, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data.get("result", data)
