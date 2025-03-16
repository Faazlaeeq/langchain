from email import message
import os
from pydoc import doc
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage,HumanMessage

load_dotenv()

current_dir= os.path.dirname(os.path.abspath(__file__))
book_path=os.path.join(current_dir,'books','islamic_banking.txt')
db_dir=os.path.join(current_dir,'db')
persitant_dir=os.path.join(db_dir,'oneoff_db')

if not os.path.exists(persitant_dir):
    
    if not os.path.exists(book_path):
        raise FileNotFoundError()
  
    print("--Loading document")  
    loader=TextLoader(book_path,encoding='utf-8')
    loaded_docs=loader.load()
    
    print("--Splitting document")
    splitter=CharacterTextSplitter(chunk_size=1000)
    documents=splitter.split_documents(loaded_docs)
    
    print("--Initiating Embedder")
    embedder=OpenAIEmbeddings(model='text-embedding-3-small')
    
    print("--Creating database")
    db=Chroma.from_documents(documents,embedder,persist_directory=persitant_dir)
    
    print("Creating Vectorstore complete")
else:
    print("VectorStore Already Exists")
    
embedder=OpenAIEmbeddings(model='text-embedding-3-small')
db=Chroma(persist_directory=persitant_dir,embedding_function=embedder)

retriever=db.as_retriever(search_type='similarity',search_kwargs={'k':1})
query="What is document title and author of this document?"
docs=retriever.invoke(query)

for i,doc in enumerate(docs):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata['source']}\n")

model=ChatOpenAI(model='gpt-4o')


combinedMessage=("Answer below given question exactly from to text given below."+"\n\n".join(query)+"\n\nText:".join([doc.page_content for doc in docs])+"\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'.")

messages=[
    SystemMessage("You are a helpful AI assistant"),
    HumanMessage(combinedMessage)
]

res=model.invoke(messages)
print(f"Model Response:\n\n{res.content}")

