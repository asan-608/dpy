import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from gensim.models import Word2Vec
import jieba
import re
import pickle
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

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

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nModel Evaluation Metrics:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # 计算相对误差
    relative_errors = np.abs(y_true - y_pred) / y_true
    mean_relative_error = np.mean(relative_errors)
    median_relative_error = np.median(relative_errors)
    
    print(f"Mean Relative Error: {mean_relative_error:.2%}")
    print(f"Median Relative Error: {median_relative_error:.2%}")

def train_model(X, y):
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # LightGBM参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'max_depth': 12,
        'min_data_in_leaf': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, y_train)
    valid_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
    # 回调函数
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[valid_data],
        callbacks=callbacks
    )
    
    # 评估模型
    val_pred = model.predict(X_val)
    evaluate_model(y_val, val_pred)
    
    return model

def main():
    try:
        # 加载数据
        print("Loading data...")
        with open('fixed_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # 处理价格
        df['price'] = df['price'].apply(lambda x: float(x.replace('¥', '').strip()))
        
        # 特征提取
        print("\nExtracting features...")
        feature_extractor = FeatureExtractor()
        feature_extractor.train(df['name'].values)
        X = feature_extractor.extract_features(df['name'].values)
        y = df['price'].values
        
        # 输出一些数据统计
        print("\nData Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Price range: ¥{min(y):.2f} - ¥{max(y):.2f}")
        print(f"Average price: ¥{np.mean(y):.2f}")
        print(f"Median price: ¥{np.median(y):.2f}")
        
        # 训练模型
        print("\nTraining model...")
        model = train_model(X, y)
        
        # 保存模型和特征提取器
        print("\nSaving models...")
        with open('price_predictor.pkl', 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_extractor': feature_extractor
            }, f)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()