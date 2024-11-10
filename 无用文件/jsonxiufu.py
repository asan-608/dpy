import json  
  
# 读取原始JSON文件内容（假设文件名为data.json）  
with open('book.json', 'r', encoding='utf-8') as file:  
    content = file.read()  
  
# 修复格式错误：在对象之间添加逗号  
fixed_content = content.replace('}{', '},{')  
  
# 确保数据以数组格式开始和结束（如果原始数据没有这些括号，则需要添加）  
if not fixed_content.startswith('[') or not fixed_content.endswith(']'):  
    fixed_content = '[' + fixed_content + ']'  
  
# 将修复后的内容解析为JSON对象  
try:  
    json_data = json.loads(fixed_content)  
except json.JSONDecodeError as e:  
    print(f"JSON解码错误: {e}")  
    # 如果仍然有错误，可能需要手动检查数据  
    # 这里可以添加更多的错误处理逻辑  
else:  
    # 将修复后的JSON对象写入新的文件（假设文件名为fixed_data.json）  
    with open('fixed_data.json', 'w', encoding='utf-8') as file:  
        json.dump(json_data, file, ensure_ascii=False, indent=4)  
  
    print("JSON格式已修复并保存到fixed_book.json文件中。")