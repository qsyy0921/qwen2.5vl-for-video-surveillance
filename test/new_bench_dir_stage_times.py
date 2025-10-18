# -*- coding: utf-8 -*-
"""
遍历目录下所有视频，统计各阶段耗时并写入 CSV（含视频元信息 & 散点图）
- 新增：video_frames / video_fps / video_duration_s
- 模型只加载一次（model_load_once_s）
- 预处理细分：msg_template_s / read_resize_s / pack_tensor_s / to_device_s / preprocess_s
- 视觉特征：vision_feat_wall_s（仅前向，不生成）
- 生成阶段：prefill_wall_s / decode_wall_s / total_wall_s / generate_wall_s
- 端到端：pipeline_wall_no_load_s / pipeline_wall_with_load_s
- 结束后可选生成散点图（每个阶段 vs video_frames）

运行示例：
(testv) python bench_dir_with_meta_and_plots.py \
  --dir "/home/l40/newdisk1/mfl/videosur/data/videos/UCF-Crime/train/Abuse" \
  --model_path "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct" \
  --dtype bfloat16 --fps 2.0 --max_frames 256 --max_pixels 151200 \
  --csv_out "/home/l40/newdisk1/mfl/videosur/test/ucf_abuse_stage_times_meta.csv" \
  --make_plots --plot_dir "/home/l40/newdisk1/mfl/videosur/test/plots"

依赖：
- decord（优先用于提取视频元信息；若无，则尝试 torchvision；再不行使用 ffprobe）
- matplotlib（仅在 --make_plots 时需要）
"""

import os
import csv
import time
import argparse
import traceback
from datetime import datetime
from contextlib import nullcontext
import subprocess
import shlex

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

# 使用 modelscope 版本
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 可选：画图
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# 可选：torchvision 作为元信息兜底（不解码像素帧的方式）
try:
    from torchvision.io import read_video_timestamps
    HAVE_TV = True
except Exception:
    HAVE_TV = False

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def list_videos(root, recursive=True):
    vids = []
    root = os.path.abspath(root)
    if recursive:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                ext = os.path.splitext(fn)[1].lower()
                if ext in ALLOWED_EXT:
                    vids.append(os.path.join(dp, fn))
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in ALLOWED_EXT:
                vids.append(p)
    return sorted(vids)

# ---------- 新增：视频元信息提取（多级兜底） ----------
def _parse_fps_str(s):
    """支持 '30000/1001' 或 '29.97' """
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) != 0 else float(a)
        return float(s)
    except Exception:
        return None

def probe_video_meta(video_path):
    """
    返回：frames(int), fps(float), duration_s(float)
    优先 decord；失败则 torchvision.read_video_timestamps；再失败试 ffprobe；还不行则 None
    """
    real = video_path[7:] if video_path.startswith("file://") else video_path

    # 1) decord
    try:
        import decord
        vr = decord.VideoReader(real)
        frames = len(vr)
        fps = float(vr.get_avg_fps())
        duration = frames / fps if fps and fps > 0 else None
        return frames, fps, duration
    except Exception:
        pass

    # 2) torchvision（只读时间戳，不解码像素）
    if HAVE_TV:
        try:
            ts_list, fps = read_video_timestamps(real)
            frames = len(ts_list)
            fps = float(fps) if fps else None
            duration = (frames / fps) if (fps and fps > 0) else None
            return frames, fps, duration
        except Exception:
            pass

    # 3) ffprobe
    try:
        cmd = (
            'ffprobe -v error -select_streams v:0 '
            '-show_entries stream=nb_frames,avg_frame_rate,duration '
            '-of default=nokey=1:noprint_wrappers=1 '
            + shlex.quote(real)
        )
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        lines = [x.strip() for x in out.splitlines() if x.strip()]
        # 可能输出顺序：nb_frames, avg_frame_rate, duration
        nb_frames = None
        fps = None
        duration = None
        for val in lines:
            # 尝试按顺序解析
            if nb_frames is None and val.isdigit():
                nb_frames = int(val)
                continue
            if fps is None:
                fps_try = _parse_fps_str(val)
                if fps_try is not None:
                    fps = fps_try
                    continue
            if duration is None:
                try:
                    duration = float(val)
                except Exception:
                    duration = None

        # 一些封装只给 duration 和 fps，缺 nb_frames；尝试 frames = round(duration*fps)
        if nb_frames is None and duration and fps and fps > 0:
            nb_frames = int(round(duration * fps))

        return nb_frames, fps, duration
    except Exception:
        pass

    return None, None, None

def build_messages(video_path, prompt, fps, min_frames, max_frames, max_pixels):
    uri = video_path if video_path.startswith("file://") else f"file://{video_path}"
    return [{
        "role": "user",
        "content": [
            {"type": "video",
             "video": uri,
             "fps": fps,
             "min_frames": min_frames,
             "max_frames": max_frames,
             "max_pixels": max_pixels},
            {"type": "text", "text": prompt},
        ],
    }]

def stage_preprocess(processor, video_path, prompt, fps, min_frames, max_frames, max_pixels, device="cuda"):
    # 1) 模板
    t0 = time.perf_counter()
    msgs = build_messages(video_path, prompt, fps, min_frames, max_frames, max_pixels)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    t1 = time.perf_counter()
    msg_template_s = t1 - t0

    # 2) 读取/抽帧/缩放
    t2 = time.perf_counter()
    image_inputs, video_inputs = process_vision_info(msgs)
    t3 = time.perf_counter()
    read_resize_s = t3 - t2

    # 3) 打包张量
    t4 = time.perf_counter()
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    t5 = time.perf_counter()
    pack_tensor_s = t5 - t4

    # 4) 拷到设备
    to_device_s = 0.0
    if device == "cuda" and torch.cuda.is_available():
        t6 = time.perf_counter()
        inputs = inputs.to("cuda")
        sync_cuda()
        t7 = time.perf_counter()
        to_device_s = t7 - t6

    preprocess_s = msg_template_s + read_resize_s + pack_tensor_s + to_device_s
    return inputs, {
        "msg_template_s": msg_template_s,
        "read_resize_s": read_resize_s,
        "pack_tensor_s": pack_tensor_s,
        "to_device_s": to_device_s,
        "preprocess_s": preprocess_s,
    }

class FirstTokenTimer(StoppingCriteria):
    """定位首个新 token 时间，用于分离 prefill/decode 墙钟时间"""
    def __init__(self, init_len, on_first_token):
        super().__init__()
        self.init_len = init_len
        self.on_first_token = on_first_token
        self.triggered = False
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if (not self.triggered) and input_ids.shape[1] > self.init_len:
            self.triggered = True
            self.on_first_token()
        return False

@torch.no_grad()
def stage_vision_features(model, inputs, dtype="bfloat16"):
    pv = inputs.get("pixel_values_videos", None)
    thw = inputs.get("video_grid_thw", None)
    if pv is None or thw is None:
        raise RuntimeError("未找到 pixel_values_videos / video_grid_thw")

    use_amp = (dtype in ("float16", "bfloat16")) and torch.cuda.is_available()
    amp_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    cm = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    sync_cuda()
    t0 = time.perf_counter()
    with cm:
        _ = model.get_video_features(pixel_values_videos=pv, video_grid_thw=thw)
    sync_cuda()
    t1 = time.perf_counter()
    return {"vision_feat_wall_s": t1 - t0}

@torch.no_grad()
def stage_generate(model, inputs, dtype="bfloat16", **gen_kwargs):
    if "input_ids" not in inputs:
        raise RuntimeError("processor 未产出 input_ids")
    init_len = inputs["input_ids"].shape[1]

    use_amp = (dtype in ("float16", "bfloat16")) and torch.cuda.is_available()
    amp_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    cm = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    prefill_wall_box = [None]
    def _first():
        prefill_wall_box[0] = time.perf_counter() - t_start

    stopping = StoppingCriteriaList([FirstTokenTimer(init_len, _first)])

    clean_gen = {k: v for k, v in gen_kwargs.items() if v is not None}

    sync_cuda()
    t_start = time.perf_counter()
    with cm:
        _ = model.generate(**inputs,
                           stopping_criteria=stopping,
                           use_cache=True,
                           return_dict_in_generate=False,
                           output_scores=False,
                           **clean_gen)
    sync_cuda()
    t_end = time.perf_counter()

    total_wall_s = t_end - t_start
    prefill_wall_s = prefill_wall_box[0] if prefill_wall_box[0] is not None else total_wall_s
    decode_wall_s = max(0.0, total_wall_s - prefill_wall_s)
    return {
        "prefill_wall_s": prefill_wall_s,
        "decode_wall_s": decode_wall_s,
        "total_wall_s": total_wall_s,
        "generate_wall_s": prefill_wall_s + decode_wall_s,
    }

def save_scatter_plots(csv_path, out_dir, stages, x_col="video_frames", title_prefix="Time vs Frames"):
    if not HAVE_MPL:
        print("⚠️ 未安装 matplotlib，跳过画图。可 pip install matplotlib")
        return
    import csv as _csv
    xs = []
    ys_map = {k: [] for k in stages}
    with open(csv_path, "r") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get("EXPLAIN_START") or row.get("EXPLAIN") or row.get("EXPLAIN_END"):
                continue
            try:
                x = int(float(row[x_col])) if row.get(x_col) not in (None, "",) else None
            except Exception:
                x = None
            if x is None:
                continue
            xs.append(x)
            for k in stages:
                try:
                    ys_map[k].append(float(row.get(k)) if row.get(k) not in (None, "") else None)
                except Exception:
                    ys_map[k].append(None)

    os.makedirs(out_dir, exist_ok=True)
    for k in stages:
        # 过滤 None
        pairs = [(x, y) for x, y in zip(xs, ys_map[k]) if (x is not None and y is not None)]
        if not pairs:
            continue
        X, Y = zip(*pairs)
        plt.figure()
        plt.scatter(X, Y, s=16)
        plt.xlabel("video_frames")
        plt.ylabel(k)
        plt.title(f"{title_prefix}: {k}")
        save_path = os.path.join(out_dir, f"scatter_{k}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"🖼 已保存散点图: {save_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="/home/l40/newdisk1/mfl/videosur/data/videos/UCF-Crime/train/Abuse")
    ap.add_argument("--recursive", action="store_true", help="递归遍历目录（默认开启）")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false")
    ap.set_defaults(recursive=True)

    ap.add_argument("--model_path", type=str, default="/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--prompt", type=str, default="Describe this video in detail.")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--min_frames", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=256)
    ap.add_argument("--max_pixels", type=int, default=360*420)

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)

    ap.add_argument("--csv_out", type=str, default="/home/l40/newdisk1/mfl/videosur/test/ucf_abuse_stage_times_meta.csv")
    ap.add_argument("--sleep_between", type=float, default=0.0)

    # OOM 降配配置
    ap.add_argument("--backoff_on_oom", action="store_true")
    ap.add_argument("--backoff_ratio", type=float, default=0.8, help="每次重试时 max_pixels/max_frames 按该比例降低")
    ap.add_argument("--min_max_pixels", type=int, default=160*224)
    ap.add_argument("--min_max_frames", type=int, default=4)
    ap.add_argument("--max_retries", type=int, default=3)

    # 绘图
    ap.add_argument("--make_plots", action="store_true", help="根据 CSV 生成散点图")
    ap.add_argument("--plot_dir", type=str, default="/home/l40/newdisk1/mfl/videosur/test/plots")

    args = ap.parse_args()

    # dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # 建议：减少碎片
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cudnn.benchmark = True

    videos = list_videos(args.dir, recursive=args.recursive)
    if not videos:
        print(f"未在目录找到视频：{args.dir}")
        return
    print(f"将处理 {len(videos)} 个视频（示例第一条）：{videos[0]}")

    # 模型加载（一次）
    print("⏳ 正在加载模型与处理器 ...")
    t0 = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch_dtype, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    sync_cuda()
    model_load_once_s = time.perf_counter() - t0
    print(f"✅ 模型加载完成，用时: {model_load_once_s:.2f} s；dtype={torch_dtype}, device_map=auto")

    # 生成参数
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=(args.temperature if args.do_sample else None),
        top_p=(args.top_p if args.do_sample else None),
        top_k=(args.top_k if args.do_sample else None),
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )

    # CSV（重新设计的大表头：先放视频元信息）
    header = [
        "timestamp","video","video_frames","video_fps","video_duration_s",
        "model_load_once_s",
        "msg_template_s","read_resize_s","pack_tensor_s","to_device_s","preprocess_s",
        "vision_feat_wall_s",
        "prefill_wall_s","decode_wall_s","total_wall_s","generate_wall_s",
        "pipeline_wall_no_load_s","pipeline_wall_with_load_s"
    ]
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ok, fail = 0, 0
    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for idx, vid in enumerate(videos, 1):
            # 新增：先探测视频元信息
            vf, vfps, vdur = probe_video_meta(vid)
            # None 做兜底
            vf_out = "" if vf is None else vf
            vfps_out = "" if vfps is None else f"{vfps:.6f}"
            vdur_out = "" if (vdur is None) else f"{vdur:.6f}"

            cur_max_pixels = int(args.max_pixels)
            cur_max_frames = int(args.max_frames)
            retries = 0

            while True:
                try:
                    # 预处理
                    inputs, pstat = stage_preprocess(
                        processor, vid, args.prompt, args.fps, args.min_frames, cur_max_frames, cur_max_pixels, device=args.device
                    )
                    # 视觉特征
                    fstat = stage_vision_features(model, inputs, dtype=args.dtype)
                    # 生成
                    gstat = stage_generate(model, inputs, dtype=args.dtype, **gen_kwargs)

                    # 汇总
                    pipeline_no_load = pstat["preprocess_s"] + fstat["vision_feat_wall_s"] + gstat["total_wall_s"]
                    pipeline_with_load = model_load_once_s + pipeline_no_load

                    w.writerow([
                        ts, vid, vf_out, vfps_out, vdur_out,
                        model_load_once_s,
                        pstat["msg_template_s"], pstat["read_resize_s"], pstat["pack_tensor_s"], pstat["to_device_s"], pstat["preprocess_s"],
                        fstat["vision_feat_wall_s"],
                        gstat["prefill_wall_s"], gstat["decode_wall_s"], gstat["total_wall_s"], gstat["generate_wall_s"],
                        pipeline_no_load, pipeline_with_load
                    ])

                    ok += 1
                    print(f"[{idx:>4}/{len(videos)}] OK  {os.path.basename(vid)}  "
                          f"frames={vf_out} pre={pstat['preprocess_s']:.3f}s feat={fstat['vision_feat_wall_s']:.3f}s gen={gstat['total_wall_s']:.3f}s")
                    break

                except RuntimeError as e:
                    msg = str(e)
                    # 识别 OOM 并按配置降配重试
                    if ("CUDA out of memory" in msg or "CUBLAS" in msg or "cudnn" in msg) and args.backoff_on_oom and retries < args.max_retries:
                        retries += 1
                        new_pixels = max(args.min_max_pixels, int(cur_max_pixels * args.backoff_ratio))
                        new_frames = max(args.min_max_frames, int(cur_max_frames * args.backoff_ratio))
                        print(f"[{idx:>4}/{len(videos)}] ⚠️ OOM：降配重试({retries}/{args.max_retries}) "
                              f"max_pixels {cur_max_pixels}→{new_pixels}, max_frames {cur_max_frames}→{new_frames}")
                        cur_max_pixels = new_pixels
                        cur_max_frames = new_frames
                        torch.cuda.empty_cache()
                        time.sleep(0.1)
                        continue
                    else:
                        fail += 1
                        print(f"[{idx:>4}/{len(videos)}] FAIL {os.path.basename(vid)} -> {e}")
                        print(traceback.format_exc())
                        break
                finally:
                    torch.cuda.empty_cache()

            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

        # EXPLAIN（列说明）
        w.writerow(["EXPLAIN_START"] + [""] * (len(header)-1))
        explain = {
            "timestamp": "写入时间(本地)",
            "video": "视频文件绝对路径",
            "video_frames": "源视频总帧数（元信息探测）",
            "video_fps": "源视频平均 FPS（元信息探测）",
            "video_duration_s": "源视频时长(秒)（元信息探测）",
            "model_load_once_s": "本次运行中模型加载一次的时间（每行重复）",
            "msg_template_s": "模板拼装(apply_chat_template)时间",
            "read_resize_s": "视频读取/抽帧/缩放(process_vision_info)时间",
            "pack_tensor_s": "processor 打包成张量时间",
            "to_device_s": "输入张量拷到GPU并同步时间",
            "preprocess_s": "预处理总时间（以上四项之和）",
            "vision_feat_wall_s": "视觉特征前向（get_video_features）墙钟时间",
            "prefill_wall_s": "生成阶段首个token前的墙钟时间",
            "decode_wall_s": "生成阶段首个token后的逐token墙钟时间",
            "total_wall_s": "生成阶段总墙钟时间（prefill+decode）",
            "generate_wall_s": "与 total_wall_s 相同，便于筛选",
            "pipeline_wall_no_load_s": "端到端(不含加载)= preprocess + vision_feat + total_wall",
            "pipeline_wall_with_load_s": "端到端(含加载)= model_load_once + pipeline_wall_no_load",
        }
        for col in header:
            row = ["EXPLAIN", col, explain.get(col, "")]
            row += [""] * (len(header) - len(row))
            w.writerow(row)
        w.writerow(["EXPLAIN_END"] + [""] * (len(header)-1))

    print(f"\n完成！成功 {ok} 个，失败 {fail} 个。CSV: {args.csv_out}")

    # 画散点图
    if args.make_plots:
        stages = [
            "msg_template_s","read_resize_s","pack_tensor_s","to_device_s","preprocess_s",
            "vision_feat_wall_s",
            "prefill_wall_s","decode_wall_s","total_wall_s",
            "pipeline_wall_no_load_s","pipeline_wall_with_load_s"
        ]
        save_scatter_plots(args.csv_out, args.plot_dir, stages, x_col="video_frames")

if __name__ == "__main__":
    main()
