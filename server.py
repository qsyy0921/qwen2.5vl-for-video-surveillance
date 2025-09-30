import time
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================== 模型路径 ==================
MODEL_PATH = "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct"

# ================== 图像分辨率控制 ==================
# 可选：按需设置图像像素范围或精确尺寸
# min_pixels = 256 * 28 * 28
# max_pixels = 1280 * 28 * 28
# processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)
# 否则使用默认：
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ================== 加载模型 ==================
print("⏳ 正在加载模型...")
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
print(f"✅ 模型加载完成，用时 {time.time() - t0:.2f} 秒")

# ================== FastAPI 应用 ==================
app = FastAPI(title="Qwen2.5-VL API Service", description="支持多模态（图像/视频）推理", version="1.0")

# ================== 请求体定义 ==================
class VisionContent(BaseModel):
    type: str  # "image" | "video" | "text"
    image: Optional[str] = None
    video: Optional[Union[str, List[str]]] = None
    text: Optional[str] = None
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None
    resized_height: Optional[int] = None
    resized_width: Optional[int] = None
    fps: Optional[float] = None

class Message(BaseModel):
    role: str
    content: List[VisionContent]

# ================== 推理函数 ==================
def run_inference(messages: List[Dict[str, Any]]):
    """
    自动检测输入类型，调用 process_vision_info，
    执行推理并返回输出与耗时。
    """
    start_t = time.time()

    # 1️⃣ 生成输入文本模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 2️⃣ 检查是否需要返回 video_kwargs
    # 如果包含 video 且不是简单 URL 列表，则 return_video_kwargs=True
    has_video = any(
        any("video" in c and c.get("video") is not None for c in msg["content"])
        for msg in messages
    )

    if has_video:
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    else:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}

    # 3️⃣ 构建输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    # 4️⃣ 推理
    gen_t0 = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    total_time = round(time.time() - start_t, 2)
    infer_time = round(time.time() - gen_t0, 2)

    return {
        "result": output_text,
        "total_time": total_time,
        "infer_time": infer_time
    }

# ================== 接口定义 ==================
@app.post("/infer")
async def infer_endpoint(request: Request):
    """
    通用推理接口
    输入: JSON messages
    输出: 模型生成文本 + 耗时
    """
    try:
        data = await request.json()
        messages = data if isinstance(data, list) else [data]
        return run_inference(messages)
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def index():
    return {"message": "✅ Qwen2.5-VL server is running on port 8000!"}
