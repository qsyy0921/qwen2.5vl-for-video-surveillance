# -*- coding: utf-8 -*-
"""
测量 Qwen2.5-VL-7B-Instruct 视频特征提取耗时（仅视觉编码，不做生成）
- 单次：模型加载时间 / 预处理时间 / 视觉编码时间
- 批量：重复 N 次，打印表格并导出 CSV
- OOM 自动降配重试（降低 max_pixels）
- 在 CSV 末尾追加“列解释”区块（EXPLAIN_START/EXPLAIN/EXPLAIN_END）
"""

import os
import time
import csv
import argparse
import statistics as stats
from datetime import datetime
import traceback

import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_messages(video_path, fps, min_frames, max_frames, max_pixels):
    video_uri = video_path if video_path.startswith("file://") else f"file://{video_path}"
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_uri,
                    "fps": fps,
                    "min_frames": min_frames,
                    "max_frames": max_frames,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]


def one_trial(model, processor, args, cur_max_pixels):
    """执行一次完整流程并返回度量结果字典；可能抛出异常（外层处理 OOM 重试）"""
    # reset 峰值显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 预处理
    t2 = time.perf_counter()
    messages = build_messages(args.video_path, args.fps, args.min_frames, args.max_frames, cur_max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if args.device == "cuda" and torch.cuda.is_available():
        inputs = inputs.to("cuda")
    sync_cuda()
    t3 = time.perf_counter()
    preprocess_s = t3 - t2

    # 关键张量
    pixel_values_videos = inputs.get("pixel_values_videos", None)
    video_grid_thw = inputs.get("video_grid_thw", None)
    if pixel_values_videos is None or video_grid_thw is None:
        raise RuntimeError("未找到 pixel_values_videos / video_grid_thw；请确认 qwen_vl_utils 与 processor 版本匹配。")

    # 视觉编码（仅前向）
    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    sync_cuda()
    t4 = time.perf_counter()
    if start_event is not None:
        start_event.record()

    with torch.no_grad():
        autocast_dtype = None
        if args.dtype in ("float16", "bfloat16") and torch.cuda.is_available():
            autocast_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        cm = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype else torch.cuda.amp.autocast(enabled=False)
        with cm:
            _ = model.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw
            )

    if end_event is not None:
        end_event.record()
        torch.cuda.synchronize()

    t5 = time.perf_counter()
    wall_forward_s = t5 - t4
    gpu_forward_s = None
    if start_event is not None and end_event is not None:
        gpu_forward_s = start_event.elapsed_time(end_event) / 1000.0  # ms->s

    # 峰值显存（MiB）
    peak_mib = None
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mib = round(peak_bytes / (1024 * 1024), 1)

    # 形状信息（可选）
    vshape = tuple(pixel_values_videos.shape) if hasattr(pixel_values_videos, "shape") else None
    thw = video_grid_thw.detach().cpu().tolist() if hasattr(video_grid_thw, "detach") else None

    # 清理中间引用，利于后续试次
    del inputs, pixel_values_videos, video_grid_thw
    torch.cuda.empty_cache()

    return {
        "preprocess_s": preprocess_s,
        "gpu_forward_s": gpu_forward_s,
        "wall_forward_s": wall_forward_s,
        "peak_mib": peak_mib,
        "max_pixels_used": cur_max_pixels,
        "pixel_values_videos_shape": vshape,
        "video_grid_thw": thw,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--video_path", type=str, default="/home/l40/newdisk1/mfl/videosur/data/videos/test.mp4")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--max_pixels", type=int, default=360 * 420)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--print_feature_shape", action="store_true")

    # 多次测试 & 导出
    parser.add_argument("--repeat", type=int, default=10, help="重复试次")
    parser.add_argument("--warmup", type=int, default=1, help="预热次数（不计入统计）")
    parser.add_argument("--sleep_between", type=float, default=0.1, help="两次试验间隔秒数")
    parser.add_argument("--csv_out", type=str, default="/home/l40/newdisk1/mfl/videosur/test/qwen25vl_bench.csv")

    # OOM 降配
    parser.add_argument("--backoff_on_oom", action="store_true", help="发生 OOM 时自动降低 max_pixels 重试")
    parser.add_argument("--min_max_pixels", type=int, default=160 * 224, help="自动降配的 max_pixels 下限")
    parser.add_argument("--backoff_ratio", type=float, default=0.8, help="每次 OOM 降配倍率（0.8=降 20%）")

    args = parser.parse_args()

    # dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # 建议开启：降低碎片
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.backends.cudnn.benchmark = True

    # ============ 模型加载 ============
    print("⏳ 正在加载模型与处理器 ...")
    t0 = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    sync_cuda()
    t1 = time.perf_counter()
    print(f"✅ 模型加载完成，用时: {t1 - t0:.2f} s；dtype={torch_dtype}, device_map=auto")

    # ============ 预热（不计入统计） ============
    warmup_runs = max(0, args.warmup)
    cur_max_pixels = int(args.max_pixels)
    for i in range(warmup_runs):
        try:
            _ = one_trial(model, processor, args, cur_max_pixels)
        except Exception as e:
            print(f"⚠️ 预热第 {i+1} 次失败：{e}")

    # ============ 正式多次测试 ============
    results = []
    for i in range(args.repeat):
        retries = 0
        while True:
            try:
                res = one_trial(model, processor, args, cur_max_pixels)
                res["run"] = i + 1
                results.append(res)
                # 打印单次结果简表（避免嵌套 f-string 的引号混淆，先格式化各列）
                prep = f"{res['preprocess_s']:.3f}" if res.get("preprocess_s") is not None else " - "
                gpu = f"{res['gpu_forward_s']:.4f}" if res.get("gpu_forward_s") is not None else " - "
                wall = f"{res['wall_forward_s']:.4f}" if res.get("wall_forward_s") is not None else " - "
                peak = f"{res['peak_mib']}" if res.get("peak_mib") is not None else " - "
                print(f"[Run {i+1:02d}] prep={prep}s  gpu={gpu}s  wall={wall}s  peak={peak}MiB  max_pixels={cur_max_pixels}")
                break
            except RuntimeError as e:
                msg = str(e)
                if "CUDA out of memory" in msg and args.backoff_on_oom:
                    retries += 1
                    if retries > 3:
                        print("❌ OOM 连续重试失败（已尝试 3 次），跳过本次。")
                        results.append({
                            "run": i + 1, "status": "oom_failed",
                            "preprocess_s": None, "gpu_forward_s": None, "wall_forward_s": None,
                            "peak_mib": None, "max_pixels_used": cur_max_pixels,
                            "pixel_values_videos_shape": None, "video_grid_thw": None
                        })
                        break
                    # 降配重试
                    new_max = int(cur_max_pixels * args.backoff_ratio)
                    if new_max < args.min_max_pixels:
                        print(f"❌ OOM，但已到达 max_pixels 下限（{args.min_max_pixels}），放弃本次。")
                        results.append({
                            "run": i + 1, "status": "oom_min_reached",
                            "preprocess_s": None, "gpu_forward_s": None, "wall_forward_s": None,
                            "peak_mib": None, "max_pixels_used": cur_max_pixels,
                            "pixel_values_videos_shape": None, "video_grid_thw": None
                        })
                        break
                    print(f"⚠️ OOM，降低 max_pixels: {cur_max_pixels} -> {new_max} 后重试（第 {retries} 次）")
                    cur_max_pixels = new_max
                    torch.cuda.empty_cache()
                    time.sleep(0.1)
                    continue
                else:
                    print(f"❌ 第 {i+1} 次失败：{e}\n{traceback.format_exc()}")
                    results.append({
                        "run": i + 1, "status": "failed",
                        "preprocess_s": None, "gpu_forward_s": None, "wall_forward_s": None,
                        "peak_mib": None, "max_pixels_used": cur_max_pixels,
                        "pixel_values_videos_shape": None, "video_grid_thw": None
                    })
                    break
        time.sleep(max(0.0, args.sleep_between))

    # ============ 统计汇总 ============
    ok_runs = [r for r in results if r.get("status", "ok") == "ok"]

    def _col(name):
        vals = [r[name] for r in ok_runs if r.get(name) is not None]
        return (stats.mean(vals), stats.pstdev(vals)) if vals else (None, None)

    mean_prep, std_prep = _col("preprocess_s")
    mean_gpu, std_gpu = _col("gpu_forward_s")
    mean_wall, std_wall = _col("wall_forward_s")

    # 终端表格
    header_line = f"{'Run':>3} | {'Prep(s)':>7} | {'GPU(s)':>7} | {'Wall(s)':>7} | {'Peak(MiB)':>9} | {'max_pixels':>10} | Status"
    print("\n" + header_line)
    print("-" * len(header_line))
    for r in results:
        prep = f"{r['preprocess_s']:.3f}" if r.get("preprocess_s") is not None else " - "
        gpu = f"{r['gpu_forward_s']:.4f}" if r.get("gpu_forward_s") is not None else " - "
        wall = f"{r['wall_forward_s']:.4f}" if r.get("wall_forward_s") is not None else " - "
        peak = f"{r['peak_mib']}" if r.get("peak_mib") is not None else " - "
        maxp = str(r.get("max_pixels_used", "-"))
        line = f"{r['run']:>3} | {prep:>7} | {gpu:>7} | {wall:>7} | {peak:>9} | {maxp:>10} | {r.get('status','-')}"
        print(line)
    print("-" * len(header_line))
    mean_prep_s = f"{mean_prep:.3f}" if mean_prep is not None else " - "
    mean_gpu_s = f"{mean_gpu:.4f}" if mean_gpu is not None else " - "
    mean_wall_s = f"{mean_wall:.4f}" if mean_wall is not None else " - "
    std_prep_s = f"{std_prep:.3f}" if std_prep is not None else " - "
    std_gpu_s = f"{std_gpu:.4f}" if std_gpu is not None else " - "
    std_wall_s = f"{std_wall:.4f}" if std_wall is not None else " - "
    print(f"MEAN | {mean_prep_s:>7} | {mean_gpu_s:>7} | {mean_wall_s:>7} | {'':>9} | {'':>10} | "
          f"(std: prep={std_prep_s}, gpu={std_gpu_s}, wall={std_wall_s})")

    # 写 CSV
    header = [
        "timestamp", "run", "status", "dtype", "fps", "min_frames", "max_frames",
        "max_pixels_used", "preprocess_s", "gpu_forward_s", "wall_forward_s",
        "peak_mib", "pixel_values_videos_shape", "video_grid_thw"
    ]

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            writer.writerow([
                ts, r["run"], r.get("status", ""),
                args.dtype, args.fps, args.min_frames, args.max_frames,
                r.get("max_pixels_used", ""),
                r.get("preprocess_s", ""), r.get("gpu_forward_s", ""), r.get("wall_forward_s", ""),
                r.get("peak_mib", ""), r.get("pixel_values_videos_shape", ""), r.get("video_grid_thw", "")
            ])
        # 追加统计
        writer.writerow([ts, "MEAN", "ok-only", args.dtype, args.fps, args.min_frames, args.max_frames,
                         "", mean_prep, mean_gpu, mean_wall, "", "", ""])
        writer.writerow([ts, "STD", "ok-only", args.dtype, args.fps, args.min_frames, args.max_frames,
                         "", std_prep, std_gpu, std_wall, "", "", ""])

        # === 追加列解释（每行与表头同列数，未用列用空串补齐） ===
        explain = {
            "timestamp": "写入时间(本地)",
            "run": "第几次/MEAN/STD",
            "status": "状态: ok/failed/oom_failed/oom_min_reached",
            "dtype": "数值精度: bfloat16/float16/float32",
            "fps": "抽帧目标FPS",
            "min_frames": "最少帧数(抽帧下限)",
            "max_frames": "最多帧数(抽帧上限)",
            "max_pixels_used": "本次使用的每帧像素上限(越大→分辨率更高→更慢/更占显存)",
            "preprocess_s": "预处理时间(秒): 解码/缩放/打包+拷贝到GPU(含同步)",
            "gpu_forward_s": "视觉编码前向的CUDA事件计时(秒)",
            "wall_forward_s": "视觉编码前向的墙钟时间(秒, 含同步)",
            "peak_mib": "峰值显存(MiB): torch.cuda.max_memory_allocated()",
            "pixel_values_videos_shape": "输入张量形状(仅记录; 第一维≈视觉token数)",
            "video_grid_thw": "特征网格[T,H,W] (token空间，而非像素)"
        }

        # 分隔行（标记解释区起始）
        writer.writerow(["EXPLAIN_START"] + [""] * (len(header) - 1))
        # 逐列写解释：["EXPLAIN", 列名, 解释, ...空]
        for col in header:
            row = ["EXPLAIN", col, explain.get(col, "")]
            row += [""] * (len(header) - len(row))
            writer.writerow(row)
        # 结束标记
        writer.writerow(["EXPLAIN_END"] + [""] * (len(header) - 1))

    print(f"\n📝 结果已保存到 CSV: {args.csv_out}")

    # 可选：打印一次形状 & 网格
    if args.print_feature_shape and ok_runs:
        sample = ok_runs[-1]
        print("\n📐 Sample Shapes")
        print("pixel_values_videos_shape:", sample.get("pixel_values_videos_shape"))
        print("video_grid_thw:", sample.get("video_grid_thw"))

    print("完成！")


if __name__ == "__main__":
    main()
