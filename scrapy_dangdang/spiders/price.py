import json
import re
import matplotlib.pyplot as plt
import seaborn as sns

# 读取并尝试解析 JSON 文件
try:
    with open('book.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
except json.JSONDecodeError:
    # 如果直接解析失败，读取文件内容为字符串
    with open('book.json', 'r', encoding='utf-8') as file:
        content = file.read()

        # 使用正则表达式提取价格数据
    price_pattern = re.compile(r'"price": "\¥([\d.]+)"')
    prices_str = price_pattern.findall(content)

# 将价格字符串转换为浮点数
prices = [float(price) for price in prices_str]

# 分析价格分布（例如，计算价格范围、平均值、中位数等）
price_min, price_max = min(prices), max(prices)
price_mean = sum(prices) / len(prices)
price_median = sorted(prices)[len(prices) // 2] if len(prices) % 2 != 0 else (sorted(prices)[len(prices) // 2 - 1] +
                                                                              sorted(prices)[len(prices) // 2]) / 2

print(f"价格范围: ¥{price_min} - ¥{price_max}")
print(f"平均价格: ¥{price_mean:.2f}")
print(f"中位数价格: ¥{price_median:.2f}")

# 可视化价格分布
plt.figure(figsize=(10, 6))
sns.histplot(prices, bins=30, kde=True)
plt.title('书籍价格分布')
plt.xlabel('价格 (¥)')
plt.ylabel('频数')
plt.grid(True)
plt.show()