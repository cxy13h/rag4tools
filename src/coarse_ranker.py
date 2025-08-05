"""
粗排服务
"""
from typing import List, Dict, Any
from collections import defaultdict
from .models import CoarseRankResult


class CoarseRanker:
    """粗排服务"""
    
    def rank_tools(self, search_results: List[Dict[str, Any]]) -> List[CoarseRankResult]:
        """
        对搜索结果进行粗排
        
        Args:
            search_results: 向量搜索结果列表
            
        Returns:
            粗排结果列表，按得分降序排列
        """
        # 按UUID分组统计
        uuid_ranks = defaultdict(list)
        
        for rank, result in enumerate(search_results):
            uuid = result.get('uuid')
            if uuid:
                uuid_ranks[uuid].append(rank + 1)  # rank从1开始
        
        # 计算每个工具的粗排得分
        coarse_results = []
        for uuid, ranks in uuid_ranks.items():
            # 使用公式: 得分 = sum(1 / (rank + 1))
            score = sum(1.0 / (rank + 1) for rank in ranks)

            coarse_result = CoarseRankResult(
                uuid=uuid,
                score=score
            )
            coarse_results.append(coarse_result)
        
        # 按得分降序排列
        coarse_results.sort(key=lambda x: x.score, reverse=True)
        
        return coarse_results
    
    def get_top_candidates(self, coarse_results: List[CoarseRankResult], top_m: int) -> List[str]:
        """
        获取Top M个候选工具的UUID列表
        
        Args:
            coarse_results: 粗排结果列表
            top_m: 返回的候选数量
            
        Returns:
            UUID列表
        """
        return [result.uuid for result in coarse_results[:top_m]]
