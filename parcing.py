import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Папка для сохранения
SAVE_DIR = "texts"
os.makedirs(SAVE_DIR, exist_ok=True)

# Страница со списком документов
BASE_URL = "https://www.regmed.ru"
LIST_URL = "https://www.regmed.ru/activity/normativnye-pravovye-akty-ls/"

headers = {"User-Agent": "Mozilla/5.0"}


def get_document_links():
    """Собирает названия и ссылки на страницы документов."""
    response = requests.get(LIST_URL, headers=headers, verify=False)
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.select("li > a[href^='https://docs.eaeunion.org']")
    return [(a.text.strip(), a["href"]) for a in links]


def get_file_links(document_page_url):
    """Находит все ссылки на файлы на странице документа."""
    response = requests.get(document_page_url, headers=headers, verify=False)
    soup = BeautifulSoup(response.content, "html.parser")
    file_links = soup.select("div.DocSearchResult_Item__FileLinks a[download]")
    return [urljoin(document_page_url, a["href"]) for a in file_links]


def download_file(file_url):
    """Скачивает файл по ссылке."""
    filename = file_url.split("/")[-1]
    filepath = os.path.join(SAVE_DIR, filename)

    if os.path.exists(filepath):
        return  # Уже скачан

    response = requests.get(file_url, headers=headers, stream=True, verify=False)
    if response.status_code == 200:
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"❌ Ошибка при скачивании: {file_url}")


if __name__ == "__main__":
    docs = get_document_links()
    for title, doc_url in tqdm(docs, desc="Обработка документов"):
        file_urls = get_file_links(doc_url)
        if not file_urls:
            print(f"⚠️ Нет файлов на странице: {doc_url}")
        for file_url in file_urls:
            download_file(file_url)
