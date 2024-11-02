import requests  
import json  
  
# URL和请求头  
url = "https://xiaoai.plus/v1/chat/completions"  
headers = {  
    "Content-Type": "application/json",  
    # API密钥  
    'Authorization': 'Bearer sk-dbhV140jGebFvQbT2B2KxfnEzZdrzQRmaYoxzMgVcErTXACh',  
}  
  
# 请求体  
payload = {  
    "messages": [  
        {  
            "role": "system",  
            "content": "你是一个大语言模型机器人"  
        },  
        {  
            "role": "user",  
            "content": "告诉我有关《卜筮正宗》这本书的有关信息，150字"  
        }  
    ],  
    "stream": False,  
    "model": "gpt-3.5-turbo",  # 穷人家的孩子只能使用3.5
    "temperature": 0.5,  
    "presence_penalty": 0,  
    "frequency_penalty": 0,  
    "top_p": 1  
}  
  
# 发送POST请求  
response = requests.post(url, headers=headers, data=json.dumps(payload))  
  
# 解析响应内容  
response_data = json.loads(response.text)  

try:  
    response_text = response_data['choices'][0]['message']['content']  
except (KeyError, IndexError, TypeError):  
    response_text = "无法从响应中提取文本内容"  
  
# 将响应内容写入到文本文件中  
with open('gpt_response.txt', 'w', encoding='utf-8') as file:  
    file.write(response_text)  
  
# 打印确认信息  
print("响应已保存到gpt_response.txt文件中")