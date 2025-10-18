from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import logging

# 设置详细的日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detailed_debug():
    model_path = "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct"
    video_path = "/home/l40/newdisk1/mfl/videosur/data/videos/demo.mp4"
    
    logger.info("开始加载模型和处理器")
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
    logger.info("模型和处理器加载完成")
    
    # 构建消息
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
                {"type": "text", "text": "描述视频的内容。"},
            ],
        }
    ]
    
    logger.info(f"构建的消息: {messages}")
    
    # 应用聊天模板
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logger.info(f"应用聊天模板后的文本: {text}")
    
    # 处理视觉信息
    logger.info("开始处理视觉信息")
    image_inputs, video_inputs = process_vision_info(messages)
    
    logger.info(f"图像输入: {type(image_inputs)}, 长度: {len(image_inputs) if image_inputs else 0}")
    logger.info(f"视频输入: {type(video_inputs)}, 长度: {len(video_inputs) if video_inputs else 0}")
    
    if video_inputs:
        for i, video in enumerate(video_inputs):
            logger.info(f"视频 {i}: 形状 {video.shape if hasattr(video, 'shape') else 'N/A'}")
    
    # 处理器调用
    logger.info("调用处理器")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 详细检查输入
    logger.info("=== 输入张量详情 ===")
    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            logger.info(f"{key}: 形状 {value.shape}, 类型 {value.dtype}")
        else:
            logger.info(f"{key}: {type(value)}")
    
    # 移动到设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    logger.info(f"输入已移动到设备: {device}")
    
    # 模型推理
    logger.info("开始模型推理")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1280)
    
    logger.info(f"生成的token IDs形状: {generated_ids.shape}")
    
    # 后处理
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    logger.info("推理完成")
    return output_text[0]

if __name__ == "__main__":
    result = detailed_debug()
    print("\n" + "="*50)
    print("最终结果:")
    print("="*50)
    print(result)