"""
数据模型定义
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import uuid


@dataclass
class ToolArg:
    """工具参数模型"""
    ArgName: str
    ArgDescription: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ArgName": self.ArgName,
            "ArgDescription": self.ArgDescription
        }


@dataclass
class Tool:
    """工具模型"""
    ToolName: str
    ToolDescription: str
    Args: List[ToolArg]
    uuid: Optional[str] = None
    
    def __post_init__(self):
        if self.uuid is None:
            self.uuid = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ToolName": self.ToolName,
            "ToolDescription": self.ToolDescription,
            "Args": [arg.to_dict() for arg in self.Args]
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], tool_uuid: Optional[str] = None) -> 'Tool':
        args = [ToolArg(**arg) for arg in data.get('Args', [])]
        return cls(
            ToolName=data['ToolName'],
            ToolDescription=data['ToolDescription'],
            Args=args,
            uuid=tool_uuid
        )


@dataclass
class ToolSlice:
    """工具切片模型"""
    uuid: str     # 对应工具的UUID
    embedding: List[float]  # 切片的向量表示
    slice_type: Optional[str] = None  # 切片类型：'overview' 或 'parameter'（可选）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "embedding": self.embedding,
            "slice_type": self.slice_type
        }


@dataclass
class SearchResult:
    """搜索结果模型"""
    tool: Tool
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool.to_dict(),
            "score": self.score,
            "rank": self.rank
        }


@dataclass
class CoarseRankResult:
    """粗排结果模型"""
    uuid: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "score": self.score
        }
