import sys

sys.path.append("../")
sys.path.append("./")

from loguru import logger
from omegaconf import OmegaConf

from core.dataloader.document_set import DocumentSet

if __name__ == "__main__":
    conf = OmegaConf.load("./config/config.yaml")
    if not conf.logging.debug:
        logger.remove()
    docset = DocumentSet(conf.data_config)
