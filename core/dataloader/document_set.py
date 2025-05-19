from pathlib import Path
from typing import List
import dill
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents.base import Document
from langchain_text_splitters.base import TextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from loguru import logger
from omegaconf import OmegaConf


class DocumentSet:
    def __init__(self, data_cfg: OmegaConf) -> None:
        self.data_cfg = data_cfg
        assert self.load_files(Path(self.data_cfg.path)), (
            "Не получилось загрузить файлы"
        )
        assert self.build_docs(self.files_links), "Не получилось построить документы"
        self.save_docs(self.splitted_data)

    def load_files(self, data_path: Path) -> List[str]:
        """_summary_

        Args:
            data_path (Path): _description_

        Returns:
            List[Document]: _description_
        """

        all_files = list(data_path.iterdir())
        self.files_links = sorted(all_files)

        logger.debug(f"Всего файлов: {len(all_files)}")

        return True

    def build_docs(self, docs_list: List[str]) -> List[Document]:
        """_summary_

        Args:
            docs_list (List[Document]): _description_

        Returns:
            List[Document]: _description_
        """
        splitted_data = []
        splitter = self._configure_splitter(self.data_cfg.splitter_config)
        logger.debug("Загружаем файлы...")
        for idx, doc_link in enumerate(docs_list, start=1):
            logger.info(f"{idx} / {len(docs_list)} - {doc_link}")

            loader = Docx2txtLoader(doc_link)

            try:
                documents = loader.load()
            except:
                documents = [Document(page_content="None", metadata={"source": "None"})]

            data = splitter.split_documents(documents)

            splitted_data.append(data)

        self.splitted_data = splitted_data

        logger.success("Загрузка завершена!")
        logger.info(f"Всего документов: {len(splitted_data)}")

        return True

    def save_docs(self, docs: list[Document]) -> None:
        save_path = self.data_cfg.save_path
        logger.debug(f"Сохраняю документы: {save_path}")
        with open(save_path, "wb") as file:
            dill.dump(self.splitted_data, file)

    def _configure_splitter(self, splitter_conf: OmegaConf) -> TextSplitter:
        return RecursiveCharacterTextSplitter(**splitter_conf)
