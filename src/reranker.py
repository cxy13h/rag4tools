"""
精排服务 - 简化版本
"""
from typing import List
import re
from .models import Tool, SearchResult
from .embedding_service import EmbeddingService
import numpy as np


class RerankerService:
    """精排服务 - 使用简化的文本相似度算法"""

    def __init__(self, top_n: int = 10):
        """
        初始化精排服务

        Args:
            top_n: 精排后返回的结果数量
        """
        self.top_n = top_n
        self.embedding_service = EmbeddingService()

    def _calculate_text_similarity(self, query: str, tool_text: str) -> float:
        """
        计算文本相似度（简化版本）
        使用关键词匹配和向量相似度结合
        """
        # 1. 关键词匹配得分
        query_words = set(re.findall(r'\w+', query.lower()))
        tool_words = set(re.findall(r'\w+', tool_text.lower()))

        if len(query_words) == 0:
            keyword_score = 0.0
        else:
            common_words = query_words.intersection(tool_words)
            keyword_score = len(common_words) / len(query_words)

        # 2. 向量相似度得分
        try:
            query_embedding = self.embedding_service.get_single_embedding(query)
            tool_embedding = self.embedding_service.get_single_embedding(tool_text)

            # 计算余弦相似度
            query_vec = np.array(query_embedding)
            tool_vec = np.array(tool_embedding)

            cosine_sim = np.dot(query_vec, tool_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(tool_vec))
            vector_score = float(cosine_sim)
        except:
            vector_score = 0.0

        # 综合得分：关键词匹配权重0.3，向量相似度权重0.7
        final_score = 0.3 * keyword_score + 0.7 * vector_score
        return final_score

    def rerank_tools(self, query: str, candidate_tools: List[Tool]) -> List[SearchResult]:
        """
        对候选工具进行精排

        Args:
            query: 用户查询
            candidate_tools: 候选工具列表

        Returns:
            精排后的搜索结果列表
        """
        if not candidate_tools:
            return []

        # 计算每个工具的相似度得分
        tool_scores = []
        for tool in candidate_tools:
            tool_text = tool.to_json()
            score = self._calculate_text_similarity(query, tool_text)
            tool_scores.append((tool, score))

        # 按得分降序排序
        tool_scores.sort(key=lambda x: x[1], reverse=True)

        # 构建搜索结果
        search_results = []
        for rank, (tool, score) in enumerate(tool_scores[:self.top_n]):
            search_result = SearchResult(
                tool=tool,
                score=score,
                rank=rank + 1
            )
            search_results.append(search_result)

        return search_results

    def get_top_k_tools(self, search_results: List[SearchResult], top_k: int) -> List[Tool]:
        """
        获取Top K个工具

        Args:
            search_results: 搜索结果列表
            top_k: 返回的工具数量

        Returns:
            Top K工具列表
        """
        return [result.tool for result in search_results[:top_k]]
