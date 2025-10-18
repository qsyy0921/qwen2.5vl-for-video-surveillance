import torch

# 清理未使用的显存缓存
torch.cuda.empty_cache()

# 释放未使用的显存并尝试强制垃圾回收
import gc
gc.collect()
