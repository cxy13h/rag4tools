"""
精排服务
"""
from typing import List
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from .models import Tool, SearchResult


class RerankerService:
    """精排服务 - 使用BAAI/bge-reranker-large模型"""

    def __init__(self, top_n: int = 10):
        """
        初始化精排服务

        Args:
            top_n: 精排后返回的结果数量
        """
        self.reranker = FlagEmbeddingReranker(
            top_n=top_n,
            model="BAAI/bge-reranker-large",
            use_fp16=False
        )

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

        # 准备文档内容 - 使用工具的完整JSON内容
        documents = [tool.to_json() for tool in candidate_tools]

        # 创建节点
        nodes = [NodeWithScore(node=TextNode(text=doc)) for doc in documents]

        # 创建查询包
        query_bundle = QueryBundle(query_str=query)

        # 执行精排
        ranked_nodes = self.reranker._postprocess_nodes(nodes, query_bundle)

        # 构建搜索结果
        search_results = []
        for rank, node_with_score in enumerate(ranked_nodes):
            # 找到对应的工具
            tool_json = node_with_score.node.get_content()
            tool = None
            for candidate_tool in candidate_tools:
                if candidate_tool.to_json() == tool_json:
                    tool = candidate_tool
                    break

            if tool:
                search_result = SearchResult(
                    tool=tool,
                    score=node_with_score.score if node_with_score.score else 0.0,
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
