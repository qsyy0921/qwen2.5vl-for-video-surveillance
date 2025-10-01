import sys, os, asyncio, time
from concurrent.futures import ThreadPoolExecutor
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vs_utils.anomaly import detect_anomaly_roi
from agents.scene_agent import SceneAgent
from agents.object_agent import ObjectAgent
from agents.integrate_agent import IntegrateAgent

VIDEO_PATH = "/home/l40/newdisk1/mfl/qwenvl/videos/demo.mp4"
MAX_CONCURRENT_GPU = 1
CACHE_TTL = 60

executor = ThreadPoolExecutor(max_workers=4)
gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPU)

session = requests.Session()
scene_agent = SceneAgent(session)
object_agent = ObjectAgent(session)
integrate_agent = IntegrateAgent(session)

cache = {"scene_result": None, "scene_time": 0}

async def call_agent(func, *args):
    async with gpu_semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, lambda: func(*args))

async def pipeline_once(video_path):
    mask = await asyncio.get_event_loop().run_in_executor(executor, detect_anomaly_roi, video_path)
    if mask is None:
        print(f"[{time.strftime('%X')}] 💤 未检测到异常，跳过推理")
        return

    now = time.time()
    if not cache["scene_result"] or now - cache["scene_time"] > CACHE_TTL:
        scene_task = asyncio.create_task(call_agent(scene_agent.analyze, video_path))
    else:
        scene_task = asyncio.create_task(asyncio.sleep(0, result=cache["scene_result"]))

    object_task = asyncio.create_task(call_agent(object_agent.analyze, video_path, mask))
    scene_result, object_result = await asyncio.gather(scene_task, object_task)

    if now - cache.get("scene_time", 0) > CACHE_TTL:
        cache["scene_result"], cache["scene_time"] = scene_result, now

    final_report = await call_agent(integrate_agent.analyze, scene_result, object_result)
    print(f"\n[{time.strftime('%X')}] 综合报告：\n{final_report}\n")

async def run_loop(video_path, interval=10):
    while True:
        try:
            await pipeline_once(video_path)
        except Exception as e:
            print("Error:", e)
        await asyncio.sleep(interval)

if __name__ == "__main__":
    asyncio.run(run_loop(VIDEO_PATH, interval=10))
