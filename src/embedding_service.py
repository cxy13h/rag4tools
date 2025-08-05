"""
向量化服务
"""
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class EmbeddingService:
    """向量化服务"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL")
        )
        self.model = os.getenv("EMBEDDING_MODEL")
        self.dimensions = 1024
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本的向量表示
        
        Args:
            texts: 文本列表，最多支持10条
            
        Returns:
            向量列表
        """
        if len(texts) > 10:
            raise ValueError("最多支持10条文本同时向量化")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            
            # 提取向量数据
            embeddings = []
            for data in response.data:
                embeddings.append(data.embedding)
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"向量化失败: {str(e)}")
    
    def get_single_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的向量表示
        
        Args:
            text: 文本内容
            
        Returns:
            向量
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0]
    
    def batch_embed_texts(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        批量向量化文本，自动分批处理
        
        Args:
            texts: 文本列表
            batch_size: 批次大小，默认10
            
        Returns:
            向量列表
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
