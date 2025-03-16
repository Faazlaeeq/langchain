import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


load_dotenv()

curr_dir= os.path.dirname(os.path.abspath(__file__))
books_dir=os.path.join(curr_dir,'books')
db_dir=os.path.join(curr_dir,'db')
presistant_dir=os.path.join(db_dir,'chroma_db_meta')


print(f"Books Dir :{books_dir}")
print(f"Presistant Dir: {presistant_dir}")


if not os.path.exists(presistant_dir):
    if not os.path.exists(books_dir):
        raise FileNotFoundError()
    
    document=[]
    
    books_list = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    
    for book in books_list:
        path= os.path.join(books_dir,book)
        loader=TextLoader(path,encoding='utf-8')
        book_docs=loader.load()
        
        for doc in book_docs:
            doc.metadata={'source': book}
            document.append(doc)


    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs=text_splitter.split_documents(document)
    
    embedder=OpenAIEmbeddings(model='text-embedding-3-small')
    
    db=Chroma.from_documents(documents=docs,embedding=embedder
                             ,persist_directory=presistant_dir)
    
    print("Finished Creating vector store with meta data")
else:
    print("Vector database already exist")
   
embedder=OpenAIEmbeddings(model='text-embedding-3-small')

db=Chroma(persist_directory=presistant_dir,embedding_function=embedder) 
query="Who was sherlock holmes?"

retriver=db.as_retriever(  search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1})



relevant_docs=retriver.invoke(query)

for i,doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata['source']}\n")
        
        