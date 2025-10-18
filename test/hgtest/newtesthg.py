from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os

# 使用本地模型路径
model_path = "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct"

# 检查模型路径是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型路径不存在: {model_path}")

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True  # 确保只使用本地文件
)

# 加载处理器
processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True
)

# 视频路径
video_path = "/home/l40/newdisk1/mfl/videosur/data/videos/demo.mp4"

# 检查视频文件是否存在
if not os.path.exists(video_path.replace("file://", "")):
    raise FileNotFoundError(f"视频文件不存在: {video_path}")

# 消息内容，包含本地视频路径和文本查询
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": f"file://{video_path}",
                "max_pixels": 360 * 420,
                "fps": 3.0,
            },
            {"type": "text", "text": "描述一下这个视频的内容。"},
        ],
    }
]

try:
    # 准备推理输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 将输入移动到GPU
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    else:
        print("警告: 未检测到GPU，使用CPU运行")

    # 推理：生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=1280)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    print("推理结果:")
    print(output_text[0])

except Exception as e:
    print(f"推理过程中出现错误: {e}")
    print("请检查:")
    print(f"1. 模型路径是否正确: {model_path}")
    print(f"2. 视频文件是否存在: {video_path}")
    print(f"3. 是否有足够的GPU内存")