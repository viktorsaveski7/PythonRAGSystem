"""
Document loader module.

Supports .txt and .pdf files from a directory, and web pages via URL.
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_documents(docs_dir: str) -> list[Document]:
    """
    Load all .txt and .pdf files from docs_dir,
    then split them into chunks suitable for embedding.
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(
            f"Documents directory '{docs_dir}' not found. "
            "Create it and add some .txt or .pdf files."
        )

    raw_docs: list[Document] = []

    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)

        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            raw_docs.extend(loader.load())

        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            raw_docs.extend(loader.load())

    if not raw_docs:
        raise ValueError(
            f"No .txt or .pdf files found in '{docs_dir}'. Add some documents first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    return splitter.split_documents(raw_docs)


def load_from_urls(urls: list[str]) -> list[Document]:
    """
    Scrape one or more web pages and split them into chunks.
    Uses requests + BeautifulSoup to extract only dish names and prices.
    """
    import requests
    import bs4

    all_docs = []

    for url in urls:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # Remove noise: nav, sidebar, footer, login forms
        for tag in soup.select("nav, header, footer, .widget_price_filter, .woocommerce-breadcrumb, form"):
            tag.decompose()

        # Try to extract product title + price pairs
        products = []
        for name_tag in soup.select(".woocommerce-loop-product__title"):
            name = name_tag.get_text(strip=True)
            price_tag = name_tag.find_next(class_="price")
            price = price_tag.get_text(strip=True) if price_tag else ""
            if name:
                products.append(f"{name} — {price}" if price else name)

        if not products:
            # Fallback: grab all visible text from main content
            main = soup.select_one("main") or soup.body
            content = main.get_text(separator="\n", strip=True) if main else ""
        else:
            content = "\n".join(products)

        if content:
            all_docs.append(Document(page_content=content, metadata={"source": url}))

    if not all_docs:
        raise ValueError("No content could be loaded from the provided URLs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    return splitter.split_documents(all_docs)
