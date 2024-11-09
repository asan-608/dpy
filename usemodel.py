import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import jieba
import re
from sklearn.preprocessing import StandardScaler

# 添加商品相关词典到jieba
product_terms = ['限定版', '正版', '套装', '全新', '限量', '珍藏', '典藏', '豪华']
for term in product_terms:
    jieba.add_word(term)

class FeatureExtractor:
    def __init__(self, vector_size=200, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.w2v_model = None
        self.scaler = StandardScaler()
        
    def clean_title(self, title):
        # 基础清洗
        title = title.strip()
        title = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', title)
        return title
    
    def extract_numerical_features(self, title):
        features = {
            'char_count': len(title),
            'word_count': len(list(jieba.cut(title))),
            'contains_number': int(bool(re.search(r'\d', title))),
            'number_count': len(re.findall(r'\d', title)),
        }
        
        # 特殊标记词
        special_terms = {
            '限定': 'limited',
            '豪华': 'deluxe',
            '珍藏': 'collectors',
            '套装': 'set',
            '正版': 'genuine',
            '典藏': 'collection',
            '限量': 'limited_edition',
            '全新': 'new'
        }
        
        for term, feature_name in special_terms.items():
            features[f'has_{feature_name}'] = int(term in title)
            
        return features
    
    def train(self, texts):
        # 分词
        tokenized_texts = [list(jieba.cut(text)) for text in texts]
        
        # 训练Word2Vec模型
        self.w2v_model = Word2Vec(
            tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        
        # 提取数值特征
        numerical_features = [self.extract_numerical_features(text) for text in texts]
        numerical_df = pd.DataFrame(numerical_features)
        
        # 标准化数值特征
        self.scaler.fit(numerical_df)
        
        return self
    
    def extract_features(self, texts):
        # Word2Vec特征
        w2v_features = []
        for text in texts:
            words = list(jieba.cut(text))
            word_vectors = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
            if word_vectors:
                text_vector = np.mean(word_vectors, axis=0)
            else:
                text_vector = np.zeros(self.vector_size)
            w2v_features.append(text_vector)
        
        # 数值特征
        numerical_features = [self.extract_numerical_features(text) for text in texts]
        numerical_df = pd.DataFrame(numerical_features)
        scaled_numerical = self.scaler.transform(numerical_df)
        
        # 组合所有特征
        combined_features = np.hstack([np.array(w2v_features), scaled_numerical])
        
        return combined_features

class PricePredictor:
    def __init__(self, model_path='price_predictor.pkl'):
        # 加载模型和特征提取器
        print("正在加载模型...")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.model = saved_data['model']
            self.feature_extractor = saved_data['feature_extractor']
        print("模型加载完成！")
    
    def predict(self, title):
        # 提取特征
        features = self.feature_extractor.extract_features([title])
        
        # 预测价格
        predicted_price = self.model.predict(features)[0]
        
        return predicted_price
    
    def predict_with_range(self, title, confidence=0.8):
        """预测价格并给出置信区间"""
        predicted_price = self.predict(title)
        
        # 简单的置信区间估计
        margin = predicted_price * 0.2  # 假设20%的误差范围
        lower_bound = predicted_price - margin
        upper_bound = predicted_price + margin
        
        return {
            'predicted_price': predicted_price,
            'lower_bound': max(0, lower_bound),
            'upper_bound': upper_bound
        }

def format_price(price):
    """格式化价格显示"""
    if price >= 100:
        return f"¥{price:.0f}"
    else:
        return f"¥{price:.2f}"

def main():
    try:
        # 初始化预测器
        predictor = PricePredictor()
        print("\n使用说明：")
        print("1. 输入商品标题，获取预测价格")
        print("2. 输入'q'退出程序")
        print("3. 预测结果会显示预测价格和可能的价格范围")
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return
    
    while True:
        # 获取用户输入
        title = input("\n请输入商品标题（输入'q'退出）: ")
        
        if title.lower() == 'q':
            break
        
        if not title.strip():
            print("请输入有效的商品标题！")
            continue
        
        try:
            # 预测价格
            result = predictor.predict_with_range(title)
            print(f"\n预测结果:")
            print(f"预测价格: {format_price(result['predicted_price'])}")
            print(f"价格范围: {format_price(result['lower_bound'])} - {format_price(result['upper_bound'])}")
            
            # 显示一些分析信息
            features = predictor.feature_extractor.extract_numerical_features(title)
            special_terms = [term for term, value in features.items() if term.startswith('has_') and value == 1]
            if special_terms:
                print("\n特征分析:")
                term_names = {
                    'has_limited': '限定版',
                    'has_deluxe': '豪华版',
                    'has_collectors': '珍藏版',
                    'has_set': '套装',
                    'has_genuine': '正版',
                    'has_collection': '典藏版',
                    'has_limited_edition': '限量版',
                    'has_new': '全新'
                }
                detected_terms = [term_names.get(term, term) for term in special_terms]
                print(f"检测到的特殊属性: {', '.join(detected_terms)}")
            
        except Exception as e:
            print(f"预测出错: {str(e)}")
            print("请尝试输入另一个商品标题")

if __name__ == "__main__":
    main()