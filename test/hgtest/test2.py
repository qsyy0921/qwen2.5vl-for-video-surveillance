from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import pdb  # Python 调试器

def debug_video_processing():
    # 配置路径
    model_path = "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct"
    video_path = "/home/l40/newdisk1/mfl/videosur/data/videos/demo.mp4"
    
    # 1. 调试点：模型加载前
    print("=== 调试点1: 开始加载模型 ===")
    pdb.set_trace()  # 在这里可以检查环境变量、路径等
    
    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        local_files_only=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True
    )
    
    # 2. 调试点：消息构建前
    print("=== 调试点2: 构建消息 ===")
    pdb.set_trace()
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]
    
    # 3. 调试点：聊天模板应用前
    print("=== 调试点3: 应用聊天模板 ===")
    pdb.set_trace()
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"生成的文本: {text}")
    
    # 4. 调试点：视觉信息处理前
    print("=== 调试点4: 处理视觉信息 ===")
    pdb.set_trace()
    
    image_inputs, video_inputs = process_vision_info(messages)
    print(f"图像输入类型: {type(image_inputs)}, 长度: {len(image_inputs) if image_inputs else 0}")
    print(f"视频输入类型: {type(video_inputs)}, 长度: {len(video_inputs) if video_inputs else 0}")
    
    # 5. 调试点：处理器调用前
    print("=== 调试点5: 调用处理器 ===")
    pdb.set_trace()
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 6. 调试点：检查处理后的输入
    print("=== 调试点6: 检查处理后的输入 ===")
    pdb.set_trace()
    print("输入张量的键:", inputs.keys())
    print("input_ids 形状:", inputs.input_ids.shape)
    print("attention_mask 形状:", inputs.attention_mask.shape)
    
    if hasattr(inputs, 'pixel_values'):
        print("pixel_values 形状:", inputs.pixel_values.shape)
    if hasattr(inputs, 'image_grid_thw'):
        print("image_grid_thw:", inputs.image_grid_thw)
    
    # 移动到设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    
    # 7. 调试点：模型推理前
    print("=== 调试点7: 开始模型推理 ===")
    pdb.set_trace()
    
    # 使用 torch.no_grad() 来节省内存
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    # 8. 调试点：后处理前
    print("=== 调试点8: 后处理 ===")
    pdb.set_trace()
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

if __name__ == "__main__":
    result = debug_video_processing()
    print("最终结果:", result)