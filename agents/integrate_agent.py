# integrate_agent.py
import requests
import json

API_URL = "http://127.0.0.1:8000/infer"

class IntegrateAgent:
    """
    整合智能体：融合场景和对象分析，输出综合报告与建议
    """
    def analyze(self, scene_result: str, object_result: str):
        summary_prompt = (
            f"请基于以下两部分内容生成一份综合报告：\n\n"
            f"1. 场景分析结果：{scene_result}\n\n"
            f"2. 对象分析结果：{object_result}\n\n"
            "要求输出：\n"
            "- 对整个事件的综合描述\n"
            "- 关键对象的作用与关系\n"
            "- 可能的异常或风险点\n"
            "- 给出合理的事故处置或决策建议"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": summary_prompt}
                ]
            }
        ]

        try:
            resp = requests.post(API_URL, json=messages)
            resp.raise_for_status()  # 捕获 HTTP 错误
            data = resp.json()
            return data.get("result", [""])[0] if isinstance(data.get("result"), list) else data
        except requests.exceptions.RequestException as e:
            return {"error": f"请求失败: {str(e)}"}
