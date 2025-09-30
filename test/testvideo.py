import os
import time  # 新增计时
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 模型路径和文件夹
model_path = "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct"
video_dir = "/home/l40/newdisk1/mfl/qwenvl/videos"
output_dir = "/home/l40/newdisk1/mfl/qwenvl/videosinfo"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

print("正在加载模型...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
print("模型加载完成！")

# 总结文件路径
summary_path = os.path.join(output_dir, "summary.txt")

# 遍历视频文件
for filename in os.listdir(video_dir):
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    video_path = os.path.join(video_dir, filename)
    print(f"\n==== 分析视频: {video_path} ====\n", flush=True)

    # 记录视频处理开始时间
    start_time = time.time()

    # 构造消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{video_path}"},
                {"type": "text", "text": "请总结一下这个视频的主要内容。"},
            ],
        }
    ]

    # 生成模型输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    print("video_inputs:", video_inputs, flush=True)
    if not video_inputs:
        print("⚠️ 视频输入为空，可能无法生成内容！", flush=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    print("开始生成文本...", flush=True)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    print("生成完成！", flush=True)

    # decode 整个生成，不切片
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    result = output_text[0].strip()
    if not result:
        result = "[生成为空，请检查视频或模型输入]"

    # 记录结束时间并计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"视频 {filename} 的分析结果：\n{result}", flush=True)
    print(f"视频 {filename} 的处理耗时：{elapsed_time:.2f} 秒", flush=True)

    # 保存到单独 txt 文件
    output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)

    # 追加到 summary 文件
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"\n==== {filename} ====\n{result}\n")
        f.write(f"处理耗时：{elapsed_time:.2f} 秒\n")

print("\n所有视频分析完成！")
