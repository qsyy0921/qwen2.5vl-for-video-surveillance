# -*- coding: utf-8 -*-
import os
import time
import math
import shutil
import traceback
import cv2
import torch
import pandas as pd

# ====== 路径配置 ======
MODEL_PATH = "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct"
VIDEO_PATH = "/home/l40/newdisk1/mfl/videosur/data/videos/car.mp4"
OUT_DIR = "/home/l40/newdisk1/mfl/videosur/test"
SAMPLED_DIR = os.path.join(OUT_DIR, "sampled_videos")
EXCEL_PATH = os.path.join(OUT_DIR, "center_benchmark.xlsx")

os.makedirs(SAMPLED_DIR, exist_ok=True)

# ====== 推理参数（尽量避免OOM） ======
# ↓ 保存成片段的视频帧率（为了文件兼容与体积）
SAVE_FPS_CAP = 15  # 保存片段不超过 15 FPS
SAVE_SIZE_MAX = 320  # 片段最大边缩到 320，减小体积

# ↓ 模型推理端的下采样（关键：避免 tokens/features 不匹配与显存爆）
INFER_FPS = 1.0            # 传给 messages 的 fps（让Qwen内部采样）
INFER_MAX_PIXELS = 360 * 420  # 参考官方示例，控制视觉tokens规模
MAX_NEW_TOKENS = 64

# ====== 片段比例：从 80% 到 50%，每次 -5% ======
RATIOS = [80, 75, 70, 65, 60, 55, 50]

# ====== 工具函数：检查 ffmpeg 存在与否 ======
def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

# ====== 根据系统情况选择容器与编码器 ======
def pick_video_output(path_no_ext: str):
    """
    返回 (out_path, fourcc)
    优先 mp4v/mp4（需要 ffmpeg），否则回退到 avi+MJPG。
    """
    if have_ffmpeg():
        return path_no_ext + ".mp4", cv2.VideoWriter_fourcc(*"mp4v")
    else:
        return path_no_ext + ".avi", cv2.VideoWriter_fourcc(*"MJPG")

# ====== 读取视频元数据 ======
def read_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        # 容错：个别编码可能给 0，这里兜底
        fps = 30.0
    return total_frames, fps, width, height

# ====== 居中裁剪片段到新文件（带编码器回退） ======
def write_center_clip(src_path, ratio_pct, save_dir):
    """
    ratio_pct: 80/75/.../50
    返回字典：{path, start_sec, dur_sec, fps, frames, note}
    """
    total_frames, src_fps, w, h = read_meta(src_path)
    duration_sec = total_frames / src_fps

    # 确定片段时间范围（取视频中间 ratio% 的时长）
    frac = ratio_pct / 100.0
    dur_sec = duration_sec * frac
    start_sec = (duration_sec - dur_sec) / 2.0
    end_sec = start_sec + dur_sec

    # 保存用的 fps（不超过上限）
    out_fps = min(src_fps, float(SAVE_FPS_CAP))

    # 目标尺寸（等比例把最长边缩到 SAVE_SIZE_MAX）
    long_edge = max(w, h)
    scale = SAVE_SIZE_MAX / long_edge if long_edge > SAVE_SIZE_MAX else 1.0
    out_w = int(round(w * scale))
    out_h = int(round(h * scale))

    # 准备输出
    base_noext = os.path.join(save_dir, f"clip_ratio{ratio_pct}")
    out_path, fourcc = pick_video_output(base_noext)

    # 打开输入、输出
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return None, f"无法打开源视频：{src_path}"

    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        # 尝试回退（如果前面就是mp4，则回退成avi；反之亦然）
        cap.release()
        if out_path.endswith(".mp4"):
            out_path = base_noext + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            out_path = base_noext + ".mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap = cv2.VideoCapture(src_path)
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            return None, "打开VideoWriter失败（mp4/avi都失败）"

    # 定位到起始帧
    start_frame = int(round(start_sec * src_fps))
    end_frame = int(round(end_sec * src_fps))
    start_frame = max(0, min(total_frames - 1, start_frame))
    end_frame = max(start_frame + 1, min(total_frames, end_frame))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    write_cnt = 0

    for fr in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        write_cnt += 1

    writer.release()
    cap.release()

    # 简单校验
    if write_cnt == 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return None, "导出片段失败（可能无帧/编码器不兼容）"

    return {
        "path": out_path,
        "start_sec": start_sec,
        "dur_sec": dur_sec,
        "fps": out_fps,
        "frames": write_cnt,
    }, "ok"

# ====== 加载模型（ModelScope 版，官方推荐） ======
def load_model_and_processor():
    from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading checkpoint shards …")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("✅ 模型加载完成！")
    return model, processor

# ====== 单次推理并计时 ======
def infer_clip(model, processor, clip_path):
    from qwen_vl_utils import process_vision_info

    # 构造 messages（只在这里设置 fps / max_pixels，避免重复）
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "file://" + clip_path,
                    "fps": float(INFER_FPS),
                    "max_pixels": int(INFER_MAX_PIXELS),
                },
                {"type": "text", "text": "请简要描述这个片段。"},
            ],
        }
    ]

    # 准备输入
    t0 = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    # pack
    t_pack0 = time.perf_counter()
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,      # 这里只传一次fps等
    ).to(model.device)
    t_pack1 = time.perf_counter()

    # 生成
    t_gen0 = time.perf_counter()
    output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    t_gen1 = time.perf_counter()

    # 解码
    t_dec0 = time.perf_counter()
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    text_out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    t_dec1 = time.perf_counter()

    return {
        "pack_sec": round(t_pack1 - t_pack0, 4),
        "gen_sec": round(t_gen1 - t_gen0, 4),
        "decode_sec": round(t_dec1 - t_dec0, 4),
        "total_sec": round(t_dec1 - t0, 4),
        "preview": text_out[:120].replace("\n", " "),
    }, None

# ====== 主流程 ======
def main():
    rows = []
    os.makedirs(OUT_DIR, exist_ok=True)

    # 先加载模型
    model, processor = load_model_and_processor()

    for r in RATIOS:
        print(f"\n▶️ 比例 {r}% —— 生成中心片段并推理")
        try:
            clip_info, note = write_center_clip(VIDEO_PATH, r, SAMPLED_DIR)
            if not clip_info:
                rows.append({
                    "ratio": r, "start_sec": None, "dur_sec": None, "fps": None, "frames": None,
                    "pack_sec": None, "gen_sec": None, "decode_sec": None, "total_sec": None,
                    "ok": False, "note": note or "write failed", "preview": ""
                })
                print("❌ 片段导出失败：", note)
                continue

            print(f"中心片段：start={clip_info['start_sec']:.2f}s, dur={clip_info['dur_sec']:.2f}s, "
                  f"fps={clip_info['fps']:.3f}, frames={clip_info['frames']}, file={clip_info['path']}")

            # 推理计时
            stats, err = infer_clip(model, processor, clip_info["path"])
            if err:
                rows.append({
                    "ratio": r, "start_sec": clip_info["start_sec"], "dur_sec": clip_info["dur_sec"],
                    "fps": clip_info["fps"], "frames": clip_info["frames"],
                    "pack_sec": None, "gen_sec": None, "decode_sec": None, "total_sec": None,
                    "ok": False, "note": err, "preview": ""
                })
                print("❌ 推理失败：", err)
                continue

            rows.append({
                "ratio": r,
                "start_sec": round(clip_info["start_sec"], 6),
                "dur_sec": round(clip_info["dur_sec"], 6),
                "fps": round(clip_info["fps"], 3),
                "frames": clip_info["frames"],
                **stats,
                "ok": True,
                "note": "ok",
            })
            print(f"⏱ pack={stats['pack_sec']}s  gen={stats['gen_sec']}s  decode={stats['decode_sec']}s  total={stats['total_sec']}s")
            print(f"📝 预览：{rows[-1]['preview']}")

        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            rows.append({
                "ratio": r, "start_sec": None, "dur_sec": None, "fps": None, "frames": None,
                "pack_sec": None, "gen_sec": None, "decode_sec": None, "total_sec": None,
                "ok": False, "note": f"CUDA OOM: {oom}", "preview": ""
            })
            print("❌ OOM：", str(oom))
        except Exception as e:
            rows.append({
                "ratio": r, "start_sec": None, "dur_sec": None, "fps": None, "frames": None,
                "pack_sec": None, "gen_sec": None, "decode_sec": None, "total_sec": None,
                "ok": False, "note": f"Error: {e}", "preview": ""
            })
            print("❌ 异常：", str(e))
            print(traceback.format_exc())

    # 保存 Excel
    df = pd.DataFrame(rows, columns=[
        "ratio", "start_sec", "dur_sec", "fps", "frames",
        "pack_sec", "gen_sec", "decode_sec", "total_sec",
        "ok", "note", "preview"
    ])
    df.to_excel(EXCEL_PATH, index=False)
    print("\n✅ 已保存 Excel：", EXCEL_PATH)
    print("📁 采样视频目录：", SAMPLED_DIR)

if __name__ == "__main__":
    main()
