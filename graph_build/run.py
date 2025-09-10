import os
import sys

from graph_build.config import Config
from graph_build.recipe_greap_gen import RecipeKnowledgeGraphBuilder, KimiRecipeParser

config = Config()

def main():
    """主函数"""
    print("🍳 AI菜谱知识图谱生成器")
    print("=" * 50)

    # 加载配置

    # 设置API密钥
    api_key = config.api_key
    # 获取菜谱目录
    # recipe_dir = get_recipe_directory()
    # recipe_dir = "../../../data/C8/cook"
    recipe_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "C8",
        "cook")
    print(f"菜谱目录: {recipe_dir}")

    # 确认参数
    print(f"\n配置信息:")
    print(f"- API密钥: {api_key[:8]}...")
    print(f"- 菜谱目录: {recipe_dir}")
    print(f"- 输出格式: {config.output_format}")
    print(f"- 输出目录: {config.output_dir}")

    confirm = input("\n确认开始处理? (y/N): ").strip().lower()
    if confirm != 'y':
        print("取消处理")
        return

    try:
        # 创建AI agent
        print("\n🤖 初始化AI Agent...")
        ai_agent = KimiRecipeParser(api_key, config.base_url)

        # 创建知识图谱构建器
        output_dir = config.output_dir
        batch_size = 20  # 默认批次大小为20
        print(f"- 批次大小: {batch_size}")
        builder = RecipeKnowledgeGraphBuilder(ai_agent, output_dir, batch_size)

        # 批量处理菜谱
        print(f"\n📚 开始处理菜谱目录...")
        processed, failed = builder.batch_process_recipes(recipe_dir)

        print(f"处理结果: 成功 {processed} 个，失败 {failed} 个")

        # 导出数据
        output_dir = config.output_dir
        output_format = config.output_format

        print(f"导出数据 (格式: {output_format})...")

        if output_format == "neo4j":
            builder.export_to_neo4j_csv(output_dir)
            print(f"Neo4j文件已生成: {output_dir}")
        else:
            builder.export_to_csv(output_dir)
            print(f"CSV文件已生成: {output_dir}")

        print("处理完成!")

    except KeyboardInterrupt:
        print(f"\n\n⏹️  用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {str(e)}")
        print(f"请检查API密钥、网络连接和菜谱文件格式")