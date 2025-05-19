import sys

sys.path.append("../")
sys.path.append("./")

from loguru import logger
from omegaconf import OmegaConf

from core.llm.model import LocalLLM

if __name__ == "__main__":
    conf = OmegaConf.load("./config/config.yaml")
    if not conf.logging.debug:
        logger.remove()
    model = LocalLLM(conf.llm.ollama_config)

    logger.debug(model.invoke("Как дела?"))
