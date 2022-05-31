import os

from pydantic import BaseSettings


class ServerSettings(BaseSettings):
    PROTOCOL: str = "http"
    HOST: str = os.environ["QSAR_HOST"]
    PORT: int = int(os.environ["QSAR_PORT"])
    URL: str = f"{PROTOCOL}://{HOST}:{PORT}"
    DEBUG_MODE: bool = bool(os.environ["DEBUG_MODE"])
    LOG_CONFIG: str = "log_config.yaml"


class Settings(ServerSettings):
    pass


settings = Settings()
