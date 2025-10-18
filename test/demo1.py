import inspect
from modelscope import AutoTokenizer, AutoProcessor

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

print("Tokenizer class:", tok.__class__)
print("Tokenizer file:", inspect.getsourcefile(tok.__class__))

print("Processor class:", proc.__class__)
print("Processor file:", inspect.getsourcefile(proc.__class__))
