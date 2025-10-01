
## 模型部署 




```markdown

脚本： server.py
命令： nohup /home/l40/anaconda3/envs/testv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

curl -X POST "http://127.0.0.1:8000/infer" \
-H "Content-Type: application/json" \
-d '[
  {
    "role": "user",
    "content": [
      {
        "type": "video",
        "video": "file:///home/l40/newdisk1/mfl/qwenvl/videos/demo.mp4",
        "max_pixels": 151200,
        "fps": 1.0
      },
      {
        "type": "text",
        "text": "Describe this video."
      }
    ]
  }
```




## agent部署 

vs_utils/anomaly.py是目前的异常检测模块 但是效率很低

