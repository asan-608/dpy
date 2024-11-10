from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import pickle
import numpy as np
from pathlib import Path

# 导入原有的特征提取器类
from usemodel import FeatureExtractor, PricePredictor

# 创建FastAPI应用
app = FastAPI(title="价格预测服务")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求模型
class PredictionRequest(BaseModel):
    title: str

# 定义响应模型
class PredictionResponse(BaseModel):
    predicted_price: float
    price_range: Dict[str, float]
    features: Dict[str, Any]

# 全局变量存储预测器实例
predictor: Optional[PricePredictor] = None

@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global predictor
    try:
        model_path = Path("price_predictor.pkl")
        if not model_path.exists():
            raise FileNotFoundError("Model file not found")
        predictor = PricePredictor(str(model_path))
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """预测商品价格"""
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 获取预测结果
        result = predictor.predict_with_range(request.title)
        
        # 获取特征分析
        features = predictor.feature_extractor.extract_numerical_features(request.title)
        
        # 转换特殊属性为中文
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
        
        special_features = {
            term_names.get(key, key): value 
            for key, value in features.items() 
            if key.startswith('has_') and value == 1
        }
        
        # 构建响应
        response = {
            "predicted_price": float(result['predicted_price']),
            "price_range": {
                "lower": float(result['lower_bound']),
                "upper": float(result['upper_bound'])
            },
            "features": {
                "special_attributes": special_features,
                "text_metrics": {
                    "char_count": features['char_count'],
                    "word_count": features['word_count'],
                    "contains_number": features['contains_number'],
                    "number_count": features['number_count']
                }
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "model_loaded": predictor is not None}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8010, reload=True)