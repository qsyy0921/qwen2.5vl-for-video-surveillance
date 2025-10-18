# -*- coding: utf-8 -*-
"""
测量 Qwen2.5-VL-7B-Instruct 视频“推理（生成）”耗时
- 预处理(读/抽帧/缩放/打包+拷GPU)
- Prefill（首个生成token前的前向；含视觉+文本条件编码）
- Decode（逐token生成）
- 输出CSV并在末尾追加列解释
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
    """
    在 generate 中，当生成序列长度首次超过初始长度(init_len)时，记录“prefill 结束时刻”
    """
    def __init__(self, init_len, on_first_token):
        super().__init__()
        self.init_len = init_len
        self.on_first_token = on_first_token
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if (not self.triggered) and input_ids.shape[1] > self.init_len:
            self.triggered = True
            self.on_first_token()
        return False  # 永不截停，仅用于计时


def one_trial(model, processor, args, cur_max_pixels, gen_kwargs):
    """执行一次完整推理，返回度量结果字典；可能抛出异常（外层处理 OOM 重试）"""
    # reset 峰值显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ========== 预处理 ==========
    t2 = time.perf_counter()
    messages = build_messages(args.video_path, args.prompt, args.fps, args.min_frames, args.max_frames, cur_max_pixels)
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

    # 校验关键字段
    if "input_ids" not in inputs:
        raise RuntimeError("processor 未产出 input_ids")
    init_len = inputs["input_ids"].shape[1]

    # 形状记录
    pixel_values_videos = inputs.get("pixel_values_videos", None)
    video_grid_thw = inputs.get("video_grid_thw", None)
    vshape = tuple(pixel_values_videos.shape) if hasattr(pixel_values_videos, "shape") else None
    thw = video_grid_thw.detach().cpu().tolist() if hasattr(video_grid_thw, "detach") else None

    # ========== 推理（prefill + decode）==========
    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    first_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    # 记录“prefill结束”的回调
    prefill_wall_s = [None]  # 用可变容器保存
    def _on_first_token():
        if first_event is not None:
            first_event.record()
            torch.cuda.synchronize()
        prefill_wall_s[0] = time.perf_counter() - t4_start

    autocast_dtype = None
    if args.dtype in ("float16", "bfloat16") and torch.cuda.is_available():
        autocast_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    stopping = StoppingCriteriaList([FirstTokenTimer(init_len, _on_first_token)])

    # 计时开始
    sync_cuda()
    t4_start = time.perf_counter()
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
    t4_end = time.perf_counter()

    total_wall_s = t4_end - t4_start
    prefill_wall_s = prefill_wall_s[0] if prefill_wall_s[0] is not None else total_wall_s  # 若极短输出也给个值
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
            # 未捕获到first_event（极短输出/某些策略下可能发生）
            total_gpu_s = start_event.elapsed_time(end_event) / 1000.0

    # 输出 & 新token数
    sequences = out.sequences
    new_tokens = int(sequences.shape[1] - init_len)
    # 文本（可选）
    text_out = processor.batch_decode(sequences[:, init_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] if args.decode_text else ""

    # 峰值显存（MiB）
    peak_mib = None
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mib = round(peak_bytes / (1024 * 1024), 1)

    # 吞吐（decode部分）
    tokps_gpu = (new_tokens / decode_gpu_s) if (decode_gpu_s and decode_gpu_s > 0) else None
    tokps_wall = (new_tokens / decode_wall_s) if (decode_wall_s and decode_wall_s > 0) else None

    # 清缓存
    del inputs, pixel_values_videos, video_grid_thw, sequences, out
    torch.cuda.empty_cache()

    return {
        "status": "ok",
        "preprocess_s": preprocess_s,
        "prefill_gpu_s": prefill_gpu_s,
        "prefill_wall_s": prefill_wall_s,
        "decode_gpu_s": decode_gpu_s,
        "decode_wall_s": decode_wall_s,
        "total_gpu_s": total_gpu_s,
        "total_wall_s": total_wall_s,
        "peak_mib": peak_mib,
        "init_len": int(init_len),
        "new_tokens": new_tokens,
        "tokps_gpu": tokps_gpu,
        "tokps_wall": tokps_wall,
        "max_pixels_used": cur_max_pixels,
        "pixel_values_videos_shape": vshape,
        "video_grid_thw": thw,
        "text_out": text_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--video_path", type=str, default="/home/l40/newdisk1/mfl/videosur/data/videos/car.mp4")
    parser.add_argument("--prompt", type=str, default="Describe this video in detail.")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--max_pixels", type=int, default=360 * 420)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--decode_text", action="store_true", help="是否把生成文本也存入CSV（可能包含逗号）")

    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    # 批量与导出
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sleep_between", type=float, default=0.1)
    parser.add_argument("--csv_out", type=str, default="/home/l40/newdisk1/mfl/videosur/test/qwen25vl_infer_bench.csv")

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
    # 去掉 None
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # 随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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

    # ============ 预热 ============
    cur_max_pixels = int(args.max_pixels)
    for i in range(max(0, args.warmup)):
        try:
            _ = one_trial(model, processor, args, cur_max_pixels, gen_kwargs)
        except Exception as e:
            print(f"⚠️ 预热第 {i+1} 次失败：{e}")

    # ============ 正式多次测试 ============
    results = []
    for i in range(args.repeat):
        retries = 0
        while True:
            try:
                res = one_trial(model, processor, args, cur_max_pixels, gen_kwargs)
                res["run"] = i + 1
                results.append(res)
                # 打印简表
                prep = f"{res['preprocess_s']:.3f}" if res.get("preprocess_s") is not None else " - "
                pf_gpu = f"{res['prefill_gpu_s']:.4f}" if res.get("prefill_gpu_s") is not None else " - "
                pf_wall = f"{res['prefill_wall_s']:.4f}" if res.get("prefill_wall_s") is not None else " - "
                dec_gpu = f"{res['decode_gpu_s']:.4f}" if res.get("decode_gpu_s") is not None else " - "
                dec_wall = f"{res['decode_wall_s']:.4f}" if res.get("decode_wall_s") is not None else " - "
                tot_gpu = f"{res['total_gpu_s']:.4f}" if res.get("total_gpu_s") is not None else " - "
                tot_wall = f"{res['total_wall_s']:.4f}" if res.get("total_wall_s") is not None else " - "
                peak = f"{res['peak_mib']}" if res.get("peak_mib") is not None else " - "
                newt = res.get("new_tokens", 0)
                print(f"[Run {i+1:02d}] prep={prep}s  prefill(gpu/wall)={pf_gpu}/{pf_wall}s  "
                      f"decode(gpu/wall)={dec_gpu}/{dec_wall}s  total(gpu/wall)={tot_gpu}/{tot_wall}s  "
                      f"new_tok={newt}  peak={peak}MiB  max_pixels={cur_max_pixels}")
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

    # ============ 统计汇总 ============
    ok = [r for r in results if r.get("status") == "ok"]
    def _mean_std(name):
        vals = [r[name] for r in ok if r.get(name) is not None]
        return (stats.mean(vals), stats.pstdev(vals)) if vals else (None, None)

    m_prep, s_prep = _mean_std("preprocess_s")
    m_pf_gpu, s_pf_gpu = _mean_std("prefill_gpu_s")
    m_pf_wall, s_pf_wall = _mean_std("prefill_wall_s")
    m_dec_gpu, s_dec_gpu = _mean_std("decode_gpu_s")
    m_dec_wall, s_dec_wall = _mean_std("decode_wall_s")
    m_tot_gpu, s_tot_gpu = _mean_std("total_gpu_s")
    m_tot_wall, s_tot_wall = _mean_std("total_wall_s")
    m_tokps_gpu, s_tokps_gpu = _mean_std("tokps_gpu")
    m_tokps_wall, s_tokps_wall = _mean_std("tokps_wall")

    # 终端小结
    print("\n==== SUMMARY (ok only) ====")
    def fmt(x, p=3): return f"{x:.{p}f}" if x is not None else "-"
    print(f"prep(s): {fmt(m_prep)} ± {fmt(s_prep)}")
    print(f"prefill gpu/wall(s): {fmt(m_pf_gpu,4)} / {fmt(m_pf_wall,4)}")
    print(f"decode  gpu/wall(s): {fmt(m_dec_gpu,4)} / {fmt(m_dec_wall,4)}   "
          f"tok/s gpu/wall: {fmt(m_tokps_gpu,1)} / {fmt(m_tokps_wall,1)}")
    print(f"total   gpu/wall(s): {fmt(m_tot_gpu,4)} / {fmt(m_tot_wall,4)}")

    # ============ 写 CSV ============
    header = [
        "timestamp","run","status","dtype","fps","min_frames","max_frames",
        "max_pixels_used","preprocess_s",
        "prefill_gpu_s","prefill_wall_s","decode_gpu_s","decode_wall_s",
        "total_gpu_s","total_wall_s","peak_mib","init_len","new_tokens",
        "tokps_gpu","tokps_wall","pixel_values_videos_shape","video_grid_thw",
        "prompt","gen_params","text_out"
    ]
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gen_params_str = f"max_new_tokens={args.max_new_tokens};do_sample={args.do_sample};temp={args.temperature};top_p={args.top_p};top_k={args.top_k};beams={args.num_beams};rep_pen={args.repetition_penalty};seed={args.seed}"
        for r in results:
            if r.get("status") != "ok":
                writer.writerow([ts, r.get("run",""), r.get("status",""), args.dtype, args.fps, args.min_frames, args.max_frames,
                                 "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", args.prompt, gen_params_str, ""])
                continue
            writer.writerow([
                ts, r["run"], r.get("status","ok"), args.dtype, args.fps, args.min_frames, args.max_frames,
                r.get("max_pixels_used",""),
                r.get("preprocess_s",""),
                r.get("prefill_gpu_s",""), r.get("prefill_wall_s",""),
                r.get("decode_gpu_s",""), r.get("decode_wall_s",""),
                r.get("total_gpu_s",""), r.get("total_wall_s",""),
                r.get("peak_mib",""), r.get("init_len",""), r.get("new_tokens",""),
                r.get("tokps_gpu",""), r.get("tokps_wall",""),
                r.get("pixel_values_videos_shape",""), r.get("video_grid_thw",""),
                args.prompt, gen_params_str,
                r.get("text_out","") if args.decode_text else ""
            ])

        # 追加统计行（ok-only）
        writer.writerow([ts,"MEAN","ok-only",args.dtype,args.fps,args.min_frames,args.max_frames,
                         "", m_prep, m_pf_gpu, m_pf_wall, m_dec_gpu, m_dec_wall, m_tot_gpu, m_tot_wall,
                         "", "", "", m_tokps_gpu, m_tokps_wall, "", "", args.prompt, gen_params_str, ""])
        writer.writerow([ts,"STD","ok-only",args.dtype,args.fps,args.min_frames,args.max_frames,
                         "", s_prep, s_pf_gpu, s_pf_wall, s_dec_gpu, s_dec_wall, s_tot_gpu, s_tot_wall,
                         "", "", "", s_tokps_gpu, s_tokps_wall, "", "", args.prompt, gen_params_str, ""])

        # === 列解释区 ===
        explain = {
            "timestamp": "写入时间(本地)",
            "run": "第几次/MEAN/STD",
            "status": "ok/failed/oom_failed/oom_min_reached",
            "dtype": "数值精度(bfloat16/float16/float32)",
            "fps": "抽帧目标FPS",
            "min_frames": "最少帧数(抽帧下限)",
            "max_frames": "最多帧数(抽帧上限)",
            "max_pixels_used": "每帧像素上限(越大→分辨率高→更慢/更占显存)",
            "preprocess_s": "预处理时间(秒): 解码/缩放/打包+拷GPU(含同步)",
            "prefill_gpu_s": "首个token前的GPU计时(秒): 条件编码/构建KV缓存",
            "prefill_wall_s": "首个token前的墙钟时间(秒)",
            "decode_gpu_s": "逐token生成阶段GPU计时(秒)",
            "decode_wall_s": "逐token生成阶段墙钟时间(秒)",
            "total_gpu_s": "prefill+decode的GPU总时长(秒)",
            "total_wall_s": "prefill+decode的墙钟总时长(秒)",
            "peak_mib": "峰值显存(MiB): max_memory_allocated()",
            "init_len": "初始文本token长度（含多模态special token）",
            "new_tokens": "生成的新token数",
            "tokps_gpu": "decode阶段GPU口径的token/s",
            "tokps_wall": "decode阶段墙钟口径的token/s",
            "pixel_values_videos_shape": "输入张量形状(仅记录; 第一维≈视觉token数)",
            "video_grid_thw": "特征网格[T,H,W](token空间而非像素)",
            "prompt": "提示词",
            "gen_params": "关键生成参数",
            "text_out": "生成文本（可选，开启 --decode_text）"
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
