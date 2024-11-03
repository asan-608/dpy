import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import re
import pickle
import os

class PricePredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.price_scaler = MinMaxScaler()
        self.model = None
        
    def preprocess_price(self, price_str):
        """将价格字符串转换为浮点数"""
        return float(price_str.replace('¥', '').strip())
    
    def preprocess_data(self, data):
        """预处理商品数据"""
        # 提取标题和价格
        titles = [item['name'].strip() for item in data]
        prices = [self.preprocess_price(item['price']) for item in data]
        
        # 向量化文本
        X = self.vectorizer.fit_transform(titles)
        # 标准化价格
        y = self.price_scaler.fit_transform(np.array(prices).reshape(-1, 1))
        
        return X.toarray(), y
    
    def build_model(self, input_dim):
        """构建神经网络模型"""
        model = Sequential([
            Dense(512, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, json_file_path, epochs=50, batch_size=32):
        """训练模型"""
        # 加载数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 预处理数据
        X, y = self.preprocess_data(data)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 构建并训练模型
        self.model = self.build_model(X_train.shape[1])
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # 评估模型
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n测试集平均绝对误差: ¥{test_mae:.2f}")
        
        return history
    
    def predict_price(self, title):
        """预测商品价格"""
        if not self.model:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        # 向量化输入标题
        title_vector = self.vectorizer.transform([title]).toarray()
        
        # 预测价格
        predicted_scaled = self.model.predict(title_vector)
        predicted_price = self.price_scaler.inverse_transform(predicted_scaled)[0][0]
        
        return predicted_price
    
    def save_model(self, folder_path='model'):
        """保存模型和相关组件"""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # 保存神经网络模型
        self.model.save(os.path.join(folder_path, 'price_model.h5'))
        
        # 保存向量器和定标器
        with open(os.path.join(folder_path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(folder_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.price_scaler, f)
    
    def load_model(self, folder_path='model'):
        """加载保存的模型和组件"""
        # 加载神经网络模型
        self.model = load_model(os.path.join(folder_path, 'price_model.h5'))
        
        # 加载向量器和定标器
        with open(os.path.join(folder_path, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(folder_path, 'scaler.pkl'), 'rb') as f:
            self.price_scaler = pickle.load(f)

# 使用示例
def main():
    # 初始化预测器
    predictor = PricePredictor()
    
    # 训练模型
    print("开始训练模型...")
    predictor.train('fixed_data.json', epochs=50)
    
    # 保存模型
    print("\n保存模型...")
    predictor.save_model()
    
    # 预测示例
    test_titles = [
        "新书 青春文学小说",
        "限量版精装珍藏版套装",
        "畅销小说实体书"
    ]
    
    print("\n预测示例:")
    for title in test_titles:
        predicted_price = predictor.predict_price(title)
        print(f"商品标题: {title}")
        print(f"预测价格: ¥{predicted_price:.2f}\n")

if __name__ == "__main__":
    main()