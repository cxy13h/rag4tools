"""
工具切片处理器
"""
from typing import List
import json
from .models import Tool, ToolSlice
from .embedding_service import EmbeddingService


class ToolSlicer:
    """工具切片处理器"""

    def __init__(self):
        self.embedding_service = EmbeddingService()

    def slice_tool(self, tool: Tool) -> List[ToolSlice]:
        """
        将工具按概览和参数维度进行切片，并直接生成向量

        Args:
            tool: 工具对象

        Returns:
            切片列表（已包含向量）
        """
        # 准备切片内容
        slice_contents = []
        slice_types = []

        # 1. 概览切片内容
        overview_content = {
            "ToolName": tool.ToolName,
            "ToolDescription": tool.ToolDescription
        }
        slice_contents.append(json.dumps(overview_content, ensure_ascii=False))
        slice_types.append("overview")

        # 2. 参数切片内容
        for arg in tool.Args:
            param_content = {
                "ArgName": arg.ArgName,
                "ArgDescription": arg.ArgDescription
            }
            slice_contents.append(json.dumps(param_content, ensure_ascii=False))
            slice_types.append("parameter")

        # 3. 批量向量化
        embeddings = self.embedding_service.batch_embed_texts(slice_contents)

        # 4. 创建切片对象
        slices = []
        for embedding, slice_type in zip(embeddings, slice_types):
            slice_obj = ToolSlice(
                uuid=tool.uuid,
                embedding=embedding,
                slice_type=slice_type
            )
            slices.append(slice_obj)

        return slices

    def slice_tools(self, tools: List[Tool]) -> List[ToolSlice]:
        """
        批量切片处理

        Args:
            tools: 工具列表

        Returns:
            所有切片的列表（已包含向量）
        """
        all_slices = []
        for tool in tools:
            slices = self.slice_tool(tool)
            all_slices.extend(slices)
        return all_slices
