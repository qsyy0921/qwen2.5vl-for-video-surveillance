import time
import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# -----------------------------
# 加载模型和处理器（只需加载一次）
# -----------------------------
model_path = "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# -----------------------------
# 构造视频消息
# -----------------------------
video_path = "file:///home/l40/newdisk1/mfl/videosur/data/videos/demo.mp4"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 151200,  # 可根据需要调整
                "fps": 1.0
            },
            {"type": "text", "text": "Describe this video."}
        ]
    }
]

# -----------------------------
# 计时：视频处理
# -----------------------------
start_video_process = time.time()
image_inputs, video_inputs = process_vision_info(messages)
end_video_process = time.time()
print(f"视频处理时间: {end_video_process - start_video_process:.2f} 秒")

# -----------------------------
# 构造模型输入
# -----------------------------
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

# -----------------------------
# 计时：模型推理生成输出
# -----------------------------
start_infer = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=128)
end_infer = time.time()
print(f"模型生成输出时间: {end_infer - start_infer:.2f} 秒")

# -----------------------------
# 解码输出
# -----------------------------
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("生成文本结果:")
print(output_text)
