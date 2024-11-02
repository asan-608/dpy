import json


def fix_json_quotes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 将单引号替换为双引号
    content = content.replace("'", '"')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


# 调用函数
fix_json_quotes('book.json')