import os
from dataclasses import dataclass


@dataclass
class Config:
    data_path: str = os.path.join(os.path.dirname(__file__), "data", "cook")
    index_save_path: str = "./vector_index"
    output_dir: str = "./ai_output"
    output_format: str = "neo4j"
    api_key = os.environ.get("MOONSHOT_API_KEY")
    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2-0711-preview"
    base_url = "https://api.moonshot.cn/v1"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

