from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    api_name: str = "Gun Detection service"
    revision: str = "local"
    od_model_path: str = "/Users/dylanjitton/Documents/topicos IA/1er parcial/1er parcial/topicos-ia-2024-1er-parcial/models/best.pt"
    seg_model_path: str = "yolov8n-seg.pt"
    log_level: str = "DEBUG"


@cache
def get_settings():
    print("getting settings...")
    return Settings()
