from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    dashscope_api_key: str = ""
    houseinsight_llm_model: str = "qwen3.6-flash"
    #: 分析规划专用模型；为空则沿用 houseinsight_llm_model（如 qwen-max）
    houseinsight_plan_model: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    #: 为 True 时恢复「清洗 ReAct Agent + 工具改表」旧路径（仅对照/排障）；生产请保持 False，用确定性 ETL
    houseinsight_legacy_agent_clean: bool = False
    #: 为 True 且配置了 API Key 时，对 description_raw 抽样调用 LLM 写入描述相关弱特征列（失败则跳过）
    houseinsight_description_enrich: bool = False
    houseinsight_description_sample_n: int = 200
    #: 分析规划 LLM 请求中「列画像」最多保留的列数，用于控 token
    houseinsight_plan_profile_max_cols: int = 48

    #: 清洗–质检 ReAct 回路：最多执行清洗的次数（含首轮）
    houseinsight_max_clean_attempts: int = 3
    quality_min_rows: int = 25
    quality_min_retention_ratio: float = 0.02
    quality_min_unit_price_coverage: float = 0.15
    quality_min_geo_coverage: float = 0.15


@lru_cache
def get_settings() -> Settings:
    return Settings()
