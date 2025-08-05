"""
RAG系统主服务
"""
from typing import List, Dict, Any
import json

from .models import Tool, ToolArg
from .slicer import ToolSlicer
from .embedding_service import EmbeddingService
from .redis_service import RedisService
from .coarse_ranker import CoarseRanker
from .reranker import RerankerService


class RAGSystem:
    """RAG系统主服务类"""
    
    def __init__(self, rerank_top_n: int = 10):
        """
        初始化RAG系统
        
        Args:
            rerank_top_n: 精排返回的最大结果数量
        """
        self.slicer = ToolSlicer()
        self.embedding_service = EmbeddingService()
        self.redis_service = RedisService()
        self.coarse_ranker = CoarseRanker()
        self.reranker = RerankerService(top_n=rerank_top_n)
    
    def index_tools(self, tools_data: List[Dict[str, Any]]):
        """
        索引工具数据（第一阶段：数据预处理）
        
        Args:
            tools_data: 工具数据列表
        """
        print("开始索引工具数据...")
        
        # 1. 解析工具数据
        tools = []
        for tool_data in tools_data:
            tool = Tool.from_dict(tool_data)
            tools.append(tool)
        
        print(f"解析了 {len(tools)} 个工具")
        
        # 2. 存储完整工具信息到Redis
        for tool in tools:
            self.redis_service.store_tool(tool)
        
        print("完整工具信息已存储到Redis")
        
        # 3. 切片处理（包含向量化）
        all_slices = self.slicer.slice_tools(tools)
        print(f"生成了 {len(all_slices)} 个切片并完成向量化")

        # 4. 存储切片到向量数据库
        self.redis_service.store_tool_slices(all_slices)
        print("切片已存储到向量数据库")
        
        print("工具索引完成！")
    
    def search_tools(self, query: str, top_n: int = 100, top_m: int = 20, top_k: int = 5) -> List[Tool]:
        """
        搜索工具（第二阶段：粗排 + 第三阶段：精排）
        
        Args:
            query: 用户查询
            top_n: 粗排阶段检索的切片数量
            top_m: 粗排后保留的候选工具数量
            top_k: 最终返回的工具数量
            
        Returns:
            Top K工具列表
        """
        print(f"开始搜索: {query}")
        
        # 1. 查询向量化
        query_embedding = self.embedding_service.get_single_embedding(query)
        print("查询向量化完成")
        
        # 2. 粗排：向量检索
        search_results = self.redis_service.search_similar_slices(query_embedding, top_n)
        print(f"检索到 {len(search_results)} 个相似切片")
        
        # 3. 粗排：去重和排序
        coarse_results = self.coarse_ranker.rank_tools(search_results)
        candidate_uuids = self.coarse_ranker.get_top_candidates(coarse_results, top_m)
        print(f"粗排后得到 {len(candidate_uuids)} 个候选工具")
        
        # 4. 获取候选工具的完整信息
        candidate_tools = self.redis_service.get_tools_by_uuids(candidate_uuids)
        print(f"获取到 {len(candidate_tools)} 个候选工具的完整信息")
        
        # 5. 精排
        rerank_results = self.reranker.rerank_tools(query, candidate_tools)
        final_tools = self.reranker.get_top_k_tools(rerank_results, top_k)
        
        print(f"精排完成，返回 {len(final_tools)} 个工具")
        
        return final_tools
    
    def clear_all_data(self):
        """清空所有数据"""
        self.redis_service.clear_all_data()
        print("所有数据已清空")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            index_info = self.redis_service.index.info()
            return {
                "index_name": index_info.get("index_name"),
                "num_docs": index_info.get("num_docs", 0),
                "vector_space_size": index_info.get("vector_space_size", 0)
            }
        except:
            return {"error": "无法获取索引信息"}


def create_sample_tools() -> List[Dict[str, Any]]:
    """创建示例工具数据"""
    return [
        {
            "ToolName": "get_stock_price",
            "ToolDescription": "用于查询指定股票代码的实时价格。",
            "Args": [
                {"ArgName": "symbol", "ArgDescription": "股票代码，例如：AAPL、MSFT。"},
                {"ArgName": "id", "ArgDescription": "股票编号"}
            ]
        },
        {
            "ToolName": "get_weather",
            "ToolDescription": "查询指定城市当前的天气状况。",
            "Args": [
                {"ArgName": "city", "ArgDescription": "城市名称，例如：北京、New York。"}
            ]
        },
        {
            "ToolName": "search_web",
            "ToolDescription": "一个通用的网络搜索工具，可以查询新闻和网页。",
            "Args": [
                {"ArgName": "query", "ArgDescription": "用户的搜索关键词。"}
            ]
        }
    ]
