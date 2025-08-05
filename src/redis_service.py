"""
Redis存储服务
"""
import os
import json
from typing import List, Dict, Any, Optional
import redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from dotenv import load_dotenv

from .models import Tool, ToolSlice

# 加载环境变量
load_dotenv()


class RedisService:
    """Redis存储和检索服务"""
    
    def __init__(self):
        # Redis连接配置
        self.redis_host = os.getenv("REDIS_HOST")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_password = os.getenv("REDOS_PASSWORD")  # 注意这里使用REDOS_PASSWORD
        
        # 构建Redis URL
        if self.redis_password:
            self.redis_url = f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        else:
            self.redis_url = f"redis://{self.redis_host}:{self.redis_port}"
        
        # 初始化Redis客户端
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=True
        )
        
        # 初始化向量索引
        self.index = None
        self._init_vector_index()
    
    def _init_vector_index(self):
        """初始化向量索引"""
        schema = IndexSchema.from_dict({
            "index": {
                "name": "tool_slices_index",
                "prefix": "tool_slices"
            },
            "fields": [
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "datatype": "float32",
                        "dims": 1024,
                        "distance_metric": "cosine",
                        "m_hnsw": 16,
                        "ef_construction": 200
                    }
                },
                {
                    "name": "uuid",
                    "type": "text"
                },
                {
                    "name": "content",
                    "type": "text"
                },
                {
                    "name": "slice_type",
                    "type": "text"
                }
            ]
        })
        
        self.index = SearchIndex(schema, redis_url=self.redis_url)
        
        # 检查索引是否存在，如果不存在则创建
        try:
            self.index.info()
        except:
            self.index.create()
    
    def store_tool(self, tool: Tool):
        """
        存储完整工具信息
        
        Args:
            tool: 工具对象
        """
        key = f"tool:{tool.uuid}"
        self.redis_client.set(key, tool.to_json())
    
    def get_tool(self, uuid: str) -> Optional[Tool]:
        """
        根据UUID获取完整工具信息
        
        Args:
            uuid: 工具UUID
            
        Returns:
            工具对象或None
        """
        key = f"tool:{uuid}"
        tool_json = self.redis_client.get(key)
        if tool_json:
            tool_data = json.loads(tool_json)
            return Tool.from_dict(tool_data, uuid)
        return None
    
    def store_tool_slices(self, slices: List[ToolSlice]):
        """
        存储工具切片到向量数据库

        Args:
            slices: 切片列表（已包含向量）
        """
        # 逐个存储切片
        for i, slice_obj in enumerate(slices):
            # 生成唯一的key
            key = f"tool_slices:{i}"

            # 存储切片数据：只需要向量、UUID和切片类型
            slice_data = {
                "embedding": json.dumps(slice_obj.embedding),  # 向量序列化为JSON字符串
                "uuid": slice_obj.uuid
            }

            # 可选：存储切片类型
            if slice_obj.slice_type:
                slice_data["slice_type"] = slice_obj.slice_type

            self.redis_client.hset(key, mapping=slice_data)
    
    def search_similar_slices(self, query_embedding: List[float], num_results: int = 100) -> List[Dict[str, Any]]:
        """
        搜索相似的切片 - 简化版本，使用余弦相似度

        Args:
            query_embedding: 查询向量
            num_results: 返回结果数量

        Returns:
            搜索结果列表
        """
        import numpy as np

        # 获取所有切片
        keys = self.redis_client.keys("tool_slices:*")
        results = []

        query_vec = np.array(query_embedding)

        for key in keys:
            slice_data = self.redis_client.hgetall(key)
            if slice_data:
                try:
                    # 反序列化向量
                    embedding = json.loads(slice_data['embedding'])
                    slice_vec = np.array(embedding)

                    # 计算余弦相似度
                    cosine_sim = np.dot(query_vec, slice_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(slice_vec))

                    result = {
                        'uuid': slice_data['uuid'],
                        'score': float(cosine_sim)
                    }

                    # 可选：添加切片类型
                    if 'slice_type' in slice_data:
                        result['slice_type'] = slice_data['slice_type']

                    results.append(result)
                except:
                    continue

        # 按相似度得分降序排序
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:num_results]
    
    def get_tools_by_uuids(self, uuids: List[str]) -> List[Tool]:
        """
        根据UUID列表批量获取工具
        
        Args:
            uuids: UUID列表
            
        Returns:
            工具列表
        """
        tools = []
        for uuid in uuids:
            tool = self.get_tool(uuid)
            if tool:
                tools.append(tool)
        return tools
    
    def clear_all_data(self):
        """清空所有数据（用于测试）"""
        # 删除所有工具数据
        tool_keys = self.redis_client.keys("tool:*")
        if tool_keys:
            self.redis_client.delete(*tool_keys)

        # 删除所有切片数据
        slice_keys = self.redis_client.keys("tool_slices:*")
        if slice_keys:
            self.redis_client.delete(*slice_keys)

        # 删除向量索引
        try:
            self.index.drop()
            self._init_vector_index()
        except:
            pass
