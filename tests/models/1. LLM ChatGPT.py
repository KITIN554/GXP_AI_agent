import sys

sys.path.append("../")
sys.path.append("./")

from loguru import logger
from omegaconf import OmegaConf

from core.llm.model import OpenAILLM

if __name__ == "__main__":
    conf = OmegaConf.load("./config/config.yaml")
    if not conf.logging.debug:
        logger.remove()
    model = OpenAILLM(conf.llm.chatgpt_config)

    logger.debug(model.invoke("Как дела?"))
