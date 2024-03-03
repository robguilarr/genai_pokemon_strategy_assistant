from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from setup_loader import SetupLoader

app_setup = SetupLoader()
base_llm, logger, global_conf, prompt_template_library = (
    app_setup.chat_openai,
    app_setup.logger,
    app_setup.global_conf,
    app_setup.prompt_template_library,
)


def _index_vector_store() -> None:
    """Void Function to create an RAG index inside a vector store from the source
    file."""
    logger.info("Loading & Splitting Source File")
    pdf_path = f"{global_conf['SOURCE_PDF_PATH']}/pokedex_tabletop_content.pdf"

    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    documents = [
        doc
        for doc in documents
        if isinstance(doc.page_content, str) and doc.page_content.split() != ""
    ]

    # Sample document has 343 pages, each Pokémon has 60 +- 10 words
    if global_conf["RECURSIVE_SPLITTER"]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100, chunk_overlap=200
        )
    else:  # Split based on the separator at the end of the page
        text_splitter = CharacterTextSplitter(
            chunk_size=1100,  # 60
            chunk_overlap=200,
            separator="^([0-9A-Z]+)\.\s([A-Z\s]+)$",
            is_separator_regex=True,
        )

    docs = text_splitter.split_documents(documents=documents)

    logger.info("Embedding Source File")
    vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings())

    logger.info("Saving Vector Store")
    vectorstore.save_local(f"{global_conf['VECTOR_STORE_PATH']}/pokedex_index_react")


if __name__ == "__main__":
    logger.info("Creating New Vector Store")
    _index_vector_store()
