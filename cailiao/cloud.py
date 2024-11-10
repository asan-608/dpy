import re
import collections
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 读取整个文件内容为一个字符串
with open('book.json', 'r', encoding='utf-8') as file:
    content = file.read()

# 使用正则表达式提取书名
pattern = r'"name": "([^"]*)"'
names = re.findall(pattern, content)

# 清洗书名：去除标点符号和多余空格，只保留中文字符（可根据需要调整）
cleaned_names = [''.join(re.findall(r'[\u4e00-\u9fff]', name)) for name in names]

# 统计词频
word_freq = collections.Counter()
for name in cleaned_names:
    words = re.findall(r'[\u4e00-\u9fff]+', name)  # 假设中文字符由空格或其他非中文字符分隔
    word_freq.update(words)

# 生成词云图
wordcloud = WordCloud(font_path='方正风雅宋简体.ttf',  # 指定中文字体路径
                      width=800,
                      height=400,
                      background_color='white').generate_from_frequencies(word_freq)

# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 关闭坐标轴
plt.show()