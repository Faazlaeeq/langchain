import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
current_dir= os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(current_dir,'books','odyssey.txt')

presistent_dir=os.path.join(current_dir, 'db', 'chroma_db')

if not os.path.exists(presistent_dir):
    print("Presistant directory does not exist, Initializing db")

    if not os.path.exists(file_path):
        raise FileNotFoundError("Presistant directory does not exist")

    loader=TextLoader(file_path,encoding="utf-8")
    documents=loader.load()
    
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs=text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("\n--- Creating embeddings ---")

    embeddings= OpenAIEmbeddings(model='text-embedding-3-small')
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")

    db=Chroma.from_documents(docs,embeddings,persist_directory=presistent_dir)

else:
    print('Vector already exist')


