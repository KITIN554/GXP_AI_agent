import sys

sys.path.append("../")
sys.path.append("./")

from loguru import logger
from omegaconf import OmegaConf

from core.embeddings.model import GPTEmbeddings

if __name__ == "__main__":
    conf = OmegaConf.load("./config/config.yaml")

    if not conf.logging.debug:
        logger.remove()

    logger.debug(conf.embeddings.chatgpt_config)

    model = GPTEmbeddings(conf.embeddings.chatgpt_config)

    logger.debug(model.embed_query("Привет!"))
    logger.debug(model.get_size_of_embeddings())
