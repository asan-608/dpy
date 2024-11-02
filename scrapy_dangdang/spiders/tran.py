import json


# 定义一个函数来将JSON文件转换为TXT文件
def convert_json_to_txt(json_file_path, txt_file_path):
    # 打开JSON文件并加载数据
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # 打开TXT文件并写入数据
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        # 如果数据是列表形式，遍历列表并写入每个条目
        if isinstance(data, list):
            for item in data:
                # 将字典转换为字符串并写入文件
                txt_file.write(str(item) + '\n')
        # 如果数据是字典形式，直接写入文件
        elif isinstance(data, dict):
            txt_file.write(str(data))


# 调用函数，转换文件
convert_json_to_txt('book.json', 'book.txt')