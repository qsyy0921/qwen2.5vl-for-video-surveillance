import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    model_path = "/home/l40/newdisk1/mfl/qwenvl/Qwen2.5-VL-7B-Instruct"
    
    print(f"🔄 Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    print("✅ Model loaded successfully!")
    
    print("\n================ Model Structure ================")
    print("Model class:", model.__class__.__name__)
    print("Model attributes and methods:")
    print([attr for attr in dir(model) if not attr.startswith('_')])
    
    if hasattr(model, 'model'):
        print("\n================ Inner Model Structure ================")
        print("Inner model class:", model.model.__class__.__name__)
        print("Inner model attributes and methods:")
        print([attr for attr in dir(model.model) if not attr.startswith('_')])
    
    if hasattr(model, 'get_encoder'):
        encoder = model.get_encoder()
        print("\n================ Encoder Structure ================")
        print("Encoder class:", encoder.__class__.__name__)
        print("Encoder attributes and methods:")
        print([attr for attr in dir(encoder) if not attr.startswith('_')])

if __name__ == "__main__":
    main()
