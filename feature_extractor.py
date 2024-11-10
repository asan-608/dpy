# feature_extractor.py
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
        tokenized_texts = [list(jieba.cut(text)) for text in texts]
        self.w2v_model = Word2Vec(
            tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        numerical_features = [self.extract_numerical_features(text) for text in texts]
        numerical_df = pd.DataFrame(numerical_features)
        self.scaler.fit(numerical_df)
        return self
    
    def extract_features(self, texts):
        w2v_features = []
        for text in texts:
            words = list(jieba.cut(text))
            word_vectors = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
            if word_vectors:
                text_vector = np.mean(word_vectors, axis=0)
            else:
                text_vector = np.zeros(self.vector_size)
            w2v_features.append(text_vector)
        
        numerical_features = [self.extract_numerical_features(text) for text in texts]
        numerical_df = pd.DataFrame(numerical_features)
        scaled_numerical = self.scaler.transform(numerical_df)
        
        combined_features = np.hstack([np.array(w2v_features), scaled_numerical])
        return combined_features