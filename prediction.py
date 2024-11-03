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
        """���۸��ַ���ת��Ϊ������"""
        try:
            # ������ֿ��ܵļ۸��ʽ
            if not price_str or price_str.strip() == '':
                return None
            # �Ƴ����з������ַ�������С���㣩
            price_cleaned = re.sub(r'[^\d.]', '', price_str)
            if price_cleaned:
                return float(price_cleaned)
            return None
        except (ValueError, TypeError):
            return None
    
    def preprocess_data(self, data):
        """Ԥ������Ʒ����"""
        # ��ȡ����ͼ۸�ͬʱ���˵���Ч����
        valid_data = []
        for item in data:
            price = self.preprocess_price(item.get('price', ''))
            if price is not None and item.get('name', '').strip():
                valid_data.append({
                    'name': item['name'].strip(),
                    'price': price
                })
        
        if not valid_data:
            raise ValueError("û����Ч�����ݿɹ�ѵ��")
        
        # ת��Ϊ�б�
        titles = [item['name'] for item in valid_data]
        prices = [item['price'] for item in valid_data]
        
        print(f"��������: {len(data)}")
        print(f"��Ч������: {len(valid_data)}")
        print(f"�۸�Χ: ?{min(prices):.2f} - ?{max(prices):.2f}")
        
        # �������ı�
        X = self.vectorizer.fit_transform(titles)
        # ��׼���۸�
        y = self.price_scaler.fit_transform(np.array(prices).reshape(-1, 1))
        
        return X.toarray(), y
    
    def build_model(self, input_dim):
        """����������ģ��"""
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
        """ѵ��ģ��"""
        try:
            # ��������
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"��ʼ��������...")
            X, y = self.preprocess_data(data)
            
            # �ָ�ѵ�����Ͳ��Լ�
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"��ʼѵ��ģ��...")
            # ������ѵ��ģ��
            self.model = self.build_model(X_train.shape[1])
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # ����ģ��
            test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"\n���Լ�ƽ���������: ?{self.price_scaler.inverse_transform([[test_mae]])[0][0]:.2f}")
            
            return history
            
        except Exception as e:
            print(f"ѵ�������г��ִ���: {str(e)}")
            raise
    
    def predict_price(self, title):
        """Ԥ����Ʒ�۸�"""
        if not self.model:
            raise ValueError("ģ����δѵ�������ȵ���train����")
            
        try:
            # �������������
            title_vector = self.vectorizer.transform([title]).toarray()
            
            # Ԥ��۸�
            predicted_scaled = self.model.predict(title_vector)
            predicted_price = self.price_scaler.inverse_transform(predicted_scaled)[0][0]
            
            return predicted_price
            
        except Exception as e:
            print(f"Ԥ������г��ִ���: {str(e)}")
            raise
    
    def save_model(self, folder_path='model'):
        """����ģ�ͺ�������"""
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            # ����������ģ��
            self.model.save(os.path.join(folder_path, 'price_model.h5'))
            
            # �����������Ͷ�����
            with open(os.path.join(folder_path, 'vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(os.path.join(folder_path, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.price_scaler, f)
                
            print(f"ģ���ѳɹ����浽 {folder_path} Ŀ¼")
            
        except Exception as e:
            print(f"����ģ��ʱ���ִ���: {str(e)}")
            raise
    
    def load_model(self, folder_path='model'):
        """���ر����ģ�ͺ����"""
        try:
            # ����������ģ��
            self.model = load_model(os.path.join(folder_path, 'price_model.h5'))
            
            # �����������Ͷ�����
            with open(os.path.join(folder_path, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(folder_path, 'scaler.pkl'), 'rb') as f:
                self.price_scaler = pickle.load(f)
                
            print("ģ�ͼ��سɹ�")
            
        except Exception as e:
            print(f"����ģ��ʱ���ִ���: {str(e)}")
            raise

def main():
    # ��ʼ��Ԥ����
    predictor = PricePredictor()
    
    try:
        # ѵ��ģ��
        print("��ʼѵ��ģ��...")
        predictor.train('fixed_data.json', epochs=50)
        
        # ����ģ��
        print("\n����ģ��...")
        predictor.save_model()
        
        # Ԥ��ʾ��
        test_titles = [
            "���� �ഺ��ѧС˵",
            "�����澫װ��ذ���װ",
            "����С˵ʵ����"
        ]
        
        print("\nԤ��ʾ��:")
        for title in test_titles:
            predicted_price = predictor.predict_price(title)
            print(f"��Ʒ����: {title}")
            print(f"Ԥ��۸�: ?{predicted_price:.2f}\n")
            
    except Exception as e:
        print(f"����ִ�г���: {str(e)}")

if __name__ == "__main__":
    main()