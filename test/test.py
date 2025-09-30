# from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import torch

# # ✅ 使用本地路径加载模型
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,  # 推荐 bfloat16，更节省显存
#     device_map="auto"            # 自动分配到可用 GPU
# )

# # ✅ 使用本地路径加载 processor
# processor = AutoProcessor.from_pretrained(
#     "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct"
# )


# # The default range for the number of visual tokens per image in the model is 4-16384.
# # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# # min_pixels = 256*28*28
# # max_pixels = 1280*28*28
# # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to("cuda")

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)



import time
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================== 配置路径 ==================
MODEL_PATH = "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct"
VIDEO_PATH = "file:///home/l40/newdisk1/mfl/videosur/videos/demo.mp4"

# ================== 模型加载 ==================
print("⏳ 正在加载模型...")
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
t1 = time.time()
print(f"✅ 模型加载完成，用时：{t1 - t0:.2f} 秒")

# ================== 构造输入 ==================
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": VIDEO_PATH,
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

print("🎬 正在准备视频输入...")
t2 = time.time()
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
t3 = time.time()
print(f"✅ 视频准备完成，用时：{t3 - t2:.2f} 秒")

# ================== 模型推理 ==================
print("🚀 正在执行模型推理...")
t4 = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
t5 = time.time()
print(f"✅ 推理完成，用时：{t5 - t4:.2f} 秒")

# ================== 打印结果 ==================
print("\n📝 模型输出：")
print(output_text)

print("\n⏱ 总耗时：{:.2f} 秒".format(t5 - t0))
