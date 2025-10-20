from ollama import chat
from ollama import ChatResponse
import os

os.environ['OLLAMA_GPU_METAL'] = '10'


response: ChatResponse = chat(model='deepseek-r1:latest', messages=[
  {
    'role': 'user',
    'content': '你是谁？',
  },
])
# 打印响应内容
print(response['message']['content'])

# 或者直接访问响应对象的字段
#print(response.message.content)