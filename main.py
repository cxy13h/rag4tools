"""
RAG4Tools 演示脚本
"""
import json
from src.rag_system import RAGSystem, create_sample_tools


def main():
    print("=== RAG4Tools 演示 ===")

    # 初始化RAG系统
    rag_system = RAGSystem(rerank_top_n=10)

    # 清空之前的数据（可选）
    print("\n1. 清空之前的数据...")
    rag_system.clear_all_data()

    # 创建示例工具数据
    print("\n2. 准备示例工具数据...")
    sample_tools = create_sample_tools()

    print("示例工具:")
    for i, tool in enumerate(sample_tools, 1):
        print(f"  {i}. {tool['ToolName']}: {tool['ToolDescription']}")

    # 索引工具数据
    print("\n3. 索引工具数据...")
    try:
        rag_system.index_tools(sample_tools)
        print("✅ 工具索引成功!")
    except Exception as e:
        print(f"❌ 工具索引失败: {e}")
        return

    # 获取系统统计信息
    print("\n4. 系统统计信息:")
    stats = rag_system.get_system_stats()
    print(f"  索引文档数: {stats.get('num_docs', 'N/A')}")

    # 测试搜索
    print("\n5. 测试搜索功能...")
    test_queries = [
        "如何查看股票价格？",
        "天气怎么样？",
        "搜索网页信息",
        "查询AAPL股票"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        try:
            results = rag_system.search_tools(query, top_k=3)
            print(f"找到 {len(results)} 个相关工具:")

            for i, tool in enumerate(results, 1):
                print(f"  {i}. {tool.ToolName}")
                print(f"     描述: {tool.ToolDescription}")
                print(f"     参数: {[arg.ArgName for arg in tool.Args]}")

        except Exception as e:
            print(f"❌ 搜索失败: {e}")

    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()
