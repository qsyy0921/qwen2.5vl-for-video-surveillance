# -*- coding: utf-8 -*-
"""
Qwen2.5-VL-7B-Instruct 视频端到端耗时测量（逐阶段精确计时 + CSV 导出）
阶段粒度：
- 模型加载：model_load_s
- 预处理四段：msg_template_s / read_resize_s / pack_tensor_s / to_device_s 以及总和 preprocess_s
- 视觉特征（仅get_video_features）：vision_feat_gpu_s / vision_feat_wall_s 及吞吐 vision_tokps_gpu
- 生成：prefill_gpu_s / prefill_wall_s / decode_gpu_s / decode_wall_s
- 生成总时长：total_gpu_s / total_wall_s
- 额外汇总：generate_wall_s（=prefill_wall_s+decode_wall_s）
            pipeline_wall_no_load_s（=preprocess_s+vision_feat_wall_s+total_wall_s）
            pipeline_wall_with_load_s（=model_load_s+pipeline_wall_no_load_s）
CSV 末尾包含完整列解释（EXPLAIN 区块）。
"""

import os
import time
import csv
import argparse
import statistics as stats
from datetime import datetime
import traceback

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

# 使用 modelscope 版本
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_messages(video_path, prompt, fps, min_frames, max_frames, max_pixels):
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
                {"type": "text", "text": prompt},
            ],
        }
    ]


class FirstTokenTimer(StoppingCriteria):
    """在 generate 过程中捕获“第一个新 token 产生”的时刻，用于划分 prefill 与 decode 阶段"""
    def __init__(self, init_len, on_first_token):
        super().__init__()
        self.init_len = init_len
        self.on_first_token = on_first_token
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if (not self.triggered) and input_ids.shape[1] > self.init_len:
            self.triggered = True
            self.on_first_token()
        return False  # 不截停，仅记录时刻


def time_preprocess(processor, args, cur_max_pixels):
    """细分预处理阶段：模板/读取缩放/打包/拷GPU，并返回 inputs + 统计信息"""
    messages = build_messages(args.video_path, args.prompt, args.fps, args.min_frames, args.max_frames, cur_max_pixels)

    t0 = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    t1 = time.perf_counter()
    msg_template_s = t1 - t0

    t2 = time.perf_counter()
    image_inputs, video_inputs = process_vision_info(messages)
    t3 = time.perf_counter()
    read_resize_s = t3 - t2

    t4 = time.perf_counter()
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    t5 = time.perf_counter()
    pack_tensor_s = t5 - t4

    to_device_s = 0.0
    if args.device == "cuda" and torch.cuda.is_available():
        t6 = time.perf_counter()
        inputs = inputs.to("cuda")
        sync_cuda()
        t7 = time.perf_counter()
        to_device_s = t7 - t6

    preprocess_s = msg_template_s + read_resize_s + pack_tensor_s + to_device_s

    # 形状/网格/视觉token数
    pixel_values_videos = inputs.get("pixel_values_videos", None)
    video_grid_thw = inputs.get("video_grid_thw", None)
    vshape = tuple(pixel_values_videos.shape) if hasattr(pixel_values_videos, "shape") else None
    thw = video_grid_thw.detach().cpu().tolist() if hasattr(video_grid_thw, "detach") else None

    vision_tokens = None
    try:
        if thw and len(thw) > 0:
            T, H, W = thw[0]
            vision_tokens = int(T) * int(H) * int(W)
    except Exception:
        pass

    return inputs, {
        "msg_template_s": msg_template_s,
        "read_resize_s": read_resize_s,
        "pack_tensor_s": pack_tensor_s,
        "to_device_s": to_device_s,
        "preprocess_s": preprocess_s,
        "pixel_values_videos_shape": vshape,
        "video_grid_thw": thw,
        "vision_tokens": vision_tokens
    }


def time_vision_features(model, inputs, dtype):
    """仅测 get_video_features，返回 GPU/Wall 时间与吞吐"""
    pixel_values_videos = inputs.get("pixel_values_videos", None)
    video_grid_thw = inputs.get("video_grid_thw", None)
    if pixel_values_videos is None or video_grid_thw is None:
        raise RuntimeError("未找到 pixel_values_videos / video_grid_thw；请确认 qwen_vl_utils 与 processor 版本匹配。")

    vision_tokens = None
    try:
        thw = video_grid_thw.detach().cpu().tolist()
        if thw and len(thw) > 0:
            T, H, W = thw[0]
            vision_tokens = int(T) * int(H) * int(W)
    except Exception:
        pass

    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    autocast_dtype = None
    if dtype in ("float16", "bfloat16") and torch.cuda.is_available():
        autocast_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    sync_cuda()
    t0 = time.perf_counter()
    if start_event is not None:
        start_event.record()

    with torch.no_grad():
        cm = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype else torch.cuda.amp.autocast(enabled=False)
        with cm:
            _ = model.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw
            )

    if end_event is not None:
        end_event.record()
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    wall_s = t1 - t0
    gpu_s = start_event.elapsed_time(end_event) / 1000.0 if (start_event and end_event) else None
    tokps_gpu = (vision_tokens / gpu_s) if (vision_tokens and gpu_s and gpu_s > 0) else None

    return gpu_s, wall_s, vision_tokens, tokps_gpu


def time_generate(model, processor, inputs, dtype, gen_kwargs):
    """测量 prefill / decode / total；返回各种时间与新 token 数、吞吐"""
    if "input_ids" not in inputs:
        raise RuntimeError("processor 未产出 input_ids")
    init_len = inputs["input_ids"].shape[1]

    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    first_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    prefill_wall_box = [None]

    def _on_first_token():
        if first_event is not None:
            first_event.record()
            torch.cuda.synchronize()
        prefill_wall_box[0] = time.perf_counter() - t_start

    autocast_dtype = None
    if dtype in ("float16", "bfloat16") and torch.cuda.is_available():
        autocast_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    stopping = StoppingCriteriaList([FirstTokenTimer(init_len, _on_first_token)])

    sync_cuda()
    t_start = time.perf_counter()
    if start_event is not None:
        start_event.record()

    with torch.no_grad():
        cm = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype else torch.cuda.amp.autocast(enabled=False)
        with cm:
            out = model.generate(
                **inputs,
                stopping_criteria=stopping,
                return_dict_in_generate=True,
                output_scores=False,
                **gen_kwargs
            )

    if end_event is not None:
        end_event.record()
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_wall_s = t_end - t_start
    prefill_wall_s = prefill_wall_box[0] if prefill_wall_box[0] is not None else total_wall_s
    decode_wall_s = max(0.0, total_wall_s - prefill_wall_s)

    prefill_gpu_s = None
    decode_gpu_s = None
    total_gpu_s = None
    if start_event is not None and end_event is not None:
        if first_event is not None and first_event.query():
            prefill_gpu_s = start_event.elapsed_time(first_event) / 1000.0
            decode_gpu_s = first_event.elapsed_time(end_event) / 1000.0
            total_gpu_s = start_event.elapsed_time(end_event) / 1000.0
        else:
            total_gpu_s = start_event.elapsed_time(end_event) / 1000.0

    sequences = out.sequences
    new_tokens = int(sequences.shape[1] - init_len)

    tokps_gpu = (new_tokens / decode_gpu_s) if (decode_gpu_s and decode_gpu_s > 0) else None
    tokps_wall = (new_tokens / decode_wall_s) if (decode_wall_s and decode_wall_s > 0) else None

    return {
        "init_len": int(init_len),
        "new_tokens": new_tokens,
        "prefill_gpu_s": prefill_gpu_s,
        "prefill_wall_s": prefill_wall_s,
        "decode_gpu_s": decode_gpu_s,
        "decode_wall_s": decode_wall_s,
        "total_gpu_s": total_gpu_s,
        "total_wall_s": total_wall_s,
        "tokps_gpu": tokps_gpu,
        "tokps_wall": tokps_wall,
        "sequences": out.sequences,  # 供可选 decode 文本使用
    }


def main():
    parser = argparse.ArgumentParser()
    # 基础路径
    parser.add_argument("--model_path", type=str, default="/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--video_path", type=str, default="/home/l40/newdisk1/mfl/videosur/data/videos/car.mp4")
    parser.add_argument("--prompt", type=str, default="Describe this video in detail.")

    # 预处理参数
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--max_pixels", type=int, default=360 * 420)

    # 精度/设备
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--decode_text", action="store_true", help="是否把生成文本也存入CSV（注意可能含逗号）")

    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    # 批量
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sleep_between", type=float, default=0.1)

    # 导出
    parser.add_argument("--csv_out", type=str, default="/home/l40/newdisk1/mfl/videosur/test/qwen25vl_full_bench.csv")

    # OOM 降配
    parser.add_argument("--backoff_on_oom", action="store_true")
    parser.add_argument("--min_max_pixels", type=int, default=160 * 224)
    parser.add_argument("--backoff_ratio", type=float, default=0.8)

    args = parser.parse_args()

    # dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # 生成配置
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        top_k=args.top_k if args.do_sample else None,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        use_cache=True,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # 随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # 减少碎片
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cudnn.benchmark = True

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    # ===== 模型加载 =====
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
    model_load_s = t1 - t0
    print(f"✅ 模型加载完成，用时: {model_load_s:.2f} s；dtype={torch_dtype}, device_map=auto")

    cur_max_pixels = int(args.max_pixels)

    # ===== 预热 =====
    for i in range(max(0, args.warmup)):
        try:
            inputs, _ = time_preprocess(processor, args, cur_max_pixels)
            _ = time_vision_features(model, inputs, args.dtype)
            _ = time_generate(model, processor, inputs, args.dtype, gen_kwargs)
        except Exception as e:
            print(f"⚠️ 预热第 {i+1} 次失败：{e}")

    # ===== 正式多次测试 =====
    results = []
    for i in range(args.repeat):
        retries = 0
        while True:
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                # —— 预处理（逐阶段）——
                inputs, pstat = time_preprocess(processor, args, cur_max_pixels)

                # —— 视觉特征 ——（可与 generate 分开看）
                feat_gpu_s, feat_wall_s, vision_tokens, vision_tokps_gpu = time_vision_features(model, inputs, args.dtype)

                # —— 生成（prefill / decode / total）——
                gstat = time_generate(model, processor, inputs, args.dtype, gen_kwargs)

                # 峰值显存
                peak_mib = None
                if torch.cuda.is_available():
                    peak_bytes = torch.cuda.max_memory_allocated()
                    peak_mib = round(peak_bytes / (1024 * 1024), 1)

                # 可选文本
                text_out = ""
                if args.decode_text:
                    try:
                        seq = gstat["sequences"]
                        init_len = gstat["init_len"]
                        text_out = processor.batch_decode(seq[:, init_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    except Exception:
                        text_out = ""

                # 清理
                if "sequences" in gstat:
                    del gstat["sequences"]
                del inputs
                torch.cuda.empty_cache()

                # —— 汇总/计算额外汇总列 —— 
                generate_wall_s = (gstat["prefill_wall_s"] or 0.0) + (gstat["decode_wall_s"] or 0.0)
                pipeline_wall_no_load_s = (pstat["preprocess_s"] or 0.0) + (feat_wall_s or 0.0) + (gstat["total_wall_s"] or 0.0)
                pipeline_wall_with_load_s = model_load_s + pipeline_wall_no_load_s

                res = {
                    "run": i + 1,
                    "status": "ok",
                    "device_name": device_name,
                    "dtype": args.dtype,
                    "seed": args.seed,

                    "fps": args.fps,
                    "min_frames": args.min_frames,
                    "max_frames": args.max_frames,
                    "max_pixels_used": cur_max_pixels,

                    # 逐阶段秒数（都会写入 CSV）
                    "model_load_s": model_load_s,
                    "msg_template_s": pstat["msg_template_s"],
                    "read_resize_s": pstat["read_resize_s"],
                    "pack_tensor_s": pstat["pack_tensor_s"],
                    "to_device_s": pstat["to_device_s"],
                    "preprocess_s": pstat["preprocess_s"],

                    "vision_feat_gpu_s": feat_gpu_s,
                    "vision_feat_wall_s": feat_wall_s,

                    "prefill_gpu_s": gstat["prefill_gpu_s"],
                    "prefill_wall_s": gstat["prefill_wall_s"],
                    "decode_gpu_s": gstat["decode_gpu_s"],
                    "decode_wall_s": gstat["decode_wall_s"],
                    "total_gpu_s": gstat["total_gpu_s"],
                    "total_wall_s": gstat["total_wall_s"],

                    # 额外汇总列（新增）
                    "generate_wall_s": generate_wall_s,
                    "pipeline_wall_no_load_s": pipeline_wall_no_load_s,
                    "pipeline_wall_with_load_s": pipeline_wall_with_load_s,

                    # 其他
                    "vision_tokens": vision_tokens if vision_tokens is not None else pstat["vision_tokens"],
                    "vision_tokps_gpu": vision_tokps_gpu,
                    "init_len": gstat["init_len"],
                    "new_tokens": gstat["new_tokens"],
                    "tokps_gpu": gstat["tokps_gpu"],
                    "tokps_wall": gstat["tokps_wall"],
                    "peak_mib": peak_mib,
                    "pixel_values_videos_shape": pstat["pixel_values_videos_shape"],
                    "video_grid_thw": pstat["video_grid_thw"],
                    "prompt": args.prompt,
                    "gen_params": f"max_new_tokens={args.max_new_tokens};do_sample={args.do_sample};temp={args.temperature};top_p={args.top_p};top_k={args.top_k};beams={args.num_beams};rep_pen={args.repetition_penalty};seed={args.seed}",
                    "text_out": text_out,
                }
                results.append(res)

                # 控制台简表
                print(
                    f"[Run {i+1:02d}] "
                    f"prep={res['preprocess_s']:.3f}s "
                    f"feat(gpu/wall)={res['vision_feat_gpu_s'] or 0:.4f}/{res['vision_feat_wall_s'] or 0:.4f}s "
                    f"prefill(gpu/wall)={(res['prefill_gpu_s'] or 0):.4f}/{(res['prefill_wall_s'] or 0):.4f}s "
                    f"decode(gpu/wall)={(res['decode_gpu_s'] or 0):.4f}/{(res['decode_wall_s'] or 0):.4f}s "
                    f"total(gpu/wall)={(res['total_gpu_s'] or 0):.4f}/{(res['total_wall_s'] or 0):.4f}s "
                    f"| generate_wall={generate_wall_s:.4f}s, pipeline_with_load={pipeline_wall_with_load_s:.4f}s "
                    f"| peak={res['peak_mib']}MiB"
                )
                break

            except RuntimeError as e:
                msg = str(e)
                if "CUDA out of memory" in msg and args.backoff_on_oom:
                    retries += 1
                    if retries > 3:
                        print("❌ OOM 连续重试失败（已尝试 3 次），跳过本次。")
                        results.append({"run": i + 1, "status": "oom_failed"})
                        break
                    new_max = int(cur_max_pixels * args.backoff_ratio)
                    if new_max < args.min_max_pixels:
                        print(f"❌ OOM，但已到达 max_pixels 下限（{args.min_max_pixels}），放弃本次。")
                        results.append({"run": i + 1, "status": "oom_min_reached"})
                        break
                    print(f"⚠️ OOM，降低 max_pixels: {cur_max_pixels} -> {new_max} 后重试（第 {retries} 次）")
                    cur_max_pixels = new_max
                    torch.cuda.empty_cache()
                    time.sleep(0.1)
                    continue
                else:
                    print(f"❌ 第 {i+1} 次失败：{e}\n{traceback.format_exc()}")
                    results.append({"run": i + 1, "status": "failed"})
                    break

        time.sleep(max(0.0, args.sleep_between))

    # ===== 统计 =====
    ok = [r for r in results if r.get("status") == "ok"]

    def mean_std(col):
        vals = [r[col] for r in ok if r.get(col) is not None]
        if not vals:
            return None, None
        return stats.mean(vals), stats.pstdev(vals)

    cols_to_stat = [
        "model_load_s",
        "msg_template_s","read_resize_s","pack_tensor_s","to_device_s","preprocess_s",
        "vision_feat_gpu_s","vision_feat_wall_s",
        "prefill_gpu_s","prefill_wall_s","decode_gpu_s","decode_wall_s",
        "total_gpu_s","total_wall_s",
        "generate_wall_s","pipeline_wall_no_load_s","pipeline_wall_with_load_s",
        "tokps_gpu","tokps_wall"
    ]
    stats_map = {c: mean_std(c) for c in cols_to_stat}

    # ===== 写 CSV =====
    header = [
        "timestamp","run","status","device_name","dtype","seed",
        "fps","min_frames","max_frames","max_pixels_used",
        # 每个阶段（秒）
        "model_load_s",
        "msg_template_s","read_resize_s","pack_tensor_s","to_device_s","preprocess_s",
        "vision_feat_gpu_s","vision_feat_wall_s","vision_tokens","vision_tokps_gpu",
        "prefill_gpu_s","prefill_wall_s","decode_gpu_s","decode_wall_s",
        "total_gpu_s","total_wall_s",
        # 额外汇总（秒）
        "generate_wall_s","pipeline_wall_no_load_s","pipeline_wall_with_load_s",
        # 生成相关
        "init_len","new_tokens","tokps_gpu","tokps_wall",
        # 资源/形状
        "peak_mib","pixel_values_videos_shape","video_grid_thw",
        # 记录
        "prompt","gen_params","text_out"
    ]
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for r in results:
            if r.get("status") != "ok":
                writer.writerow([ts, r.get("run",""), r.get("status",""), "", args.dtype, args.seed,
                                 args.fps, args.min_frames, args.max_frames, "",
                                 "", "", "", "", "", "",
                                 "", "", "", "",
                                 "", "", "", "",
                                 "", "", "",
                                 "", "", "",
                                 "", "", "", "", args.prompt,
                                 f"max_new_tokens={args.max_new_tokens};do_sample={args.do_sample};temp={args.temperature};top_p={args.top_p};top_k={args.top_k};beams={args.num_beams};rep_pen={args.repetition_penalty};seed={args.seed}",
                                 "" ])
                continue

            writer.writerow([
                ts, r["run"], r.get("status","ok"), r.get("device_name",""), r.get("dtype",""), r.get("seed",""),
                r.get("fps",""), r.get("min_frames",""), r.get("max_frames",""), r.get("max_pixels_used",""),
                # 阶段
                r.get("model_load_s",""),
                r.get("msg_template_s",""), r.get("read_resize_s",""), r.get("pack_tensor_s",""), r.get("to_device_s",""), r.get("preprocess_s",""),
                r.get("vision_feat_gpu_s",""), r.get("vision_feat_wall_s",""), r.get("vision_tokens",""), r.get("vision_tokps_gpu",""),
                r.get("prefill_gpu_s",""), r.get("prefill_wall_s",""), r.get("decode_gpu_s",""), r.get("decode_wall_s",""),
                r.get("total_gpu_s",""), r.get("total_wall_s",""),
                # 汇总
                r.get("generate_wall_s",""), r.get("pipeline_wall_no_load_s",""), r.get("pipeline_wall_with_load_s",""),
                # 生成相关
                r.get("init_len",""), r.get("new_tokens",""), r.get("tokps_gpu",""), r.get("tokps_wall",""),
                # 资源/形状
                r.get("peak_mib",""), r.get("pixel_values_videos_shape",""), r.get("video_grid_thw",""),
                # 记录
                r.get("prompt",""), r.get("gen_params",""), r.get("text_out","")
            ])

        # 追加统计（ok-only）
        def gs(c): return stats_map[c][0]
        def ss(c): return stats_map[c][1]
        mean_row = [ts,"MEAN","ok-only",device_name,args.dtype,args.seed,
                    args.fps,args.min_frames,args.max_frames,"",
                    gs("model_load_s"),
                    gs("msg_template_s"),gs("read_resize_s"),gs("pack_tensor_s"),gs("to_device_s"),gs("preprocess_s"),
                    gs("vision_feat_gpu_s"),gs("vision_feat_wall_s"),"", "",
                    gs("prefill_gpu_s"),gs("prefill_wall_s"),gs("decode_gpu_s"),gs("decode_wall_s"),
                    gs("total_gpu_s"),gs("total_wall_s"),
                    gs("generate_wall_s"),gs("pipeline_wall_no_load_s"),gs("pipeline_wall_with_load_s"),
                    "","",gs("tokps_gpu"),gs("tokps_wall"),
                    "","","", args.prompt,
                    f"max_new_tokens={args.max_new_tokens};do_sample={args.do_sample};temp={args.temperature};top_p={args.top_p};top_k={args.top_k};beams={args.num_beams};rep_pen={args.repetition_penalty};seed={args.seed}",
                    ""]
        std_row  = [ts,"STD","ok-only",device_name,args.dtype,args.seed,
                    args.fps,args.min_frames,args.max_frames,"",
                    ss("model_load_s"),
                    ss("msg_template_s"),ss("read_resize_s"),ss("pack_tensor_s"),ss("to_device_s"),ss("preprocess_s"),
                    ss("vision_feat_gpu_s"),ss("vision_feat_wall_s"),"", "",
                    ss("prefill_gpu_s"),ss("prefill_wall_s"),ss("decode_gpu_s"),ss("decode_wall_s"),
                    ss("total_gpu_s"),ss("total_wall_s"),
                    ss("generate_wall_s"),ss("pipeline_wall_no_load_s"),ss("pipeline_wall_with_load_s"),
                    "","",ss("tokps_gpu"),ss("tokps_wall"),
                    "","","", args.prompt,
                    f"max_new_tokens={args.max_new_tokens};do_sample={args.do_sample};temp={args.temperature};top_p={args.top_p};top_k={args.top_k};beams={args.num_beams};rep_pen={args.repetition_penalty};seed={args.seed}",
                    ""]
        writer.writerow(mean_row)
        writer.writerow(std_row)

        # === 列解释（EXPLAIN 区块） ===
        explain = {
            "timestamp": "写入时间(本地)",
            "run": "第几次/MEAN/STD",
            "status": "ok/failed/oom_failed/oom_min_reached",
            "device_name": "GPU 型号（或 cpu）",
            "dtype": "数值精度(bfloat16/float16/float32)",
            "seed": "随机种子（采样时影响解码）",

            "fps": "抽帧目标FPS",
            "min_frames": "最少帧数(抽帧下限)",
            "max_frames": "最多帧数(抽帧上限)",
            "max_pixels_used": "每帧像素上限(越大→分辨率更高→更慢/更占显存)",

            # —— 每个阶段（全部为“秒”）——
            "model_load_s": "模型加载时间（含权重加载与初始化）",
            "msg_template_s": "模板拼装时间(apply_chat_template)",
            "read_resize_s": "视频读取/抽帧/缩放时间(process_vision_info)",
            "pack_tensor_s": "processor 打包成张量时间",
            "to_device_s": "输入拷到GPU并同步时间",
            "preprocess_s": "预处理总时间（上述四项之和）",
            "vision_feat_gpu_s": "视觉特征前向-GPU计时(get_video_features)",
            "vision_feat_wall_s": "视觉特征前向-墙钟时间",
            "vision_tokens": "视觉 token 总数 = T×H×W",
            "vision_tokps_gpu": "视觉特征GPU口径吞吐(token/s)",
            "prefill_gpu_s": "首个token前GPU计时(构建KV+多模态条件编码)",
            "prefill_wall_s": "首个token前墙钟时间",
            "decode_gpu_s": "逐token生成阶段GPU计时",
            "decode_wall_s": "逐token生成阶段墙钟时间",
            "total_gpu_s": "prefill+decode 的 GPU 总时长",
            "total_wall_s": "prefill+decode 的墙钟总时长",

            # —— 额外汇总 ——（也是“秒”）
            "generate_wall_s": "生成阶段总墙钟= prefill_wall_s + decode_wall_s",
            "pipeline_wall_no_load_s": "端到端(不含加载)= preprocess_s + vision_feat_wall_s + total_wall_s",
            "pipeline_wall_with_load_s": "端到端(含加载)= model_load_s + pipeline_wall_no_load_s",

            # 生成相关/其他
            "init_len": "初始输入文本 token 长度（含多模态特殊符号）",
            "new_tokens": "新生成 token 数",
            "tokps_gpu": "decode 阶段 GPU 口径 token/s",
            "tokps_wall": "decode 阶段墙钟口径 token/s",
            "peak_mib": "峰值显存(MiB): torch.cuda.max_memory_allocated()",
            "pixel_values_videos_shape": "输入张量形状(仅记录; 第一维≈视觉token数)",
            "video_grid_thw": "特征网格[T,H,W](token空间，而非像素)",

            "prompt": "提示词",
            "gen_params": "关键生成参数",
            "text_out": "生成文本（可选; --decode_text 开启）"
        }

        writer.writerow(["EXPLAIN_START"] + [""] * (len(header) - 1))
        for col in header:
            row = ["EXPLAIN", col, explain.get(col, "")]
            row += [""] * (len(header) - len(row))
            writer.writerow(row)
        writer.writerow(["EXPLAIN_END"] + [""] * (len(header) - 1))

    print(f"\n📝 CSV 写入: {args.csv_out}")
    print("完成！")


if __name__ == "__main__":
    main()
