from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppBaseConfig:
    """Базовый класс для конфигурации с общими настройками"""
    case_sensitive = False
    env_file = "./settings/prod.env"
    env_file_encoding = "utf-8"
    env_nested_delimiter="__"
    extra = "ignore"

class DSConfig(BaseSettings):

    model_config = SettingsConfigDict(
        **AppBaseConfig.__dict__,
    )
    ds_api: str = Field(default=...)
    

ds_settings = DSConfig()