import os
from re import search
from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

current_dir=os.path.dirname(os.path.abspath(__file__))
db_dir=os.path.join(current_dir,'db')
persistant_dir=os.path.join(db_dir,'chroma_db_firecrawl_musk')

def create_vector_store():
    api_key=os.getenv('FIRECRAWL_API_KEY')
    if not api_key:
        raise ValueError("Api key not defined")
    
    loader= FireCrawlLoader('https://finance.yahoo.com/news/musk-tells-tesla-employees-hang-101843948.html',mode='scrape',api_key=api_key)
    docs=loader.load()
    
    print("Finised Crawling ")
    
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value,list):
                doc.metadata[key]= ",".join(map(str,value))
                
    text_splitter= CharacterTextSplitter(chunk_size=1000)
    documents=text_splitter.split_documents(docs)
    
    embedding= OpenAIEmbeddings(model="text-embedding-3-small")
    
    db=Chroma.from_documents(documents,embedding=embedding,persist_directory=persistant_dir)
    

if not os.path.exists(persistant_dir):
    create_vector_store()
else:
    
    print("Vector Store Already Exists")
    
embedding= OpenAIEmbeddings(model="text-embedding-3-small")
    
db=Chroma(persist_directory=persistant_dir,embedding_function=embedding)
    
def query_vectorstore(query):
    
    retriever=db.as_retriever(search_type="similarity",kwargs={'k':3})
    relevant_doc=retriever.invoke(query)
    
    # for i,doc in enumerate(relevant_doc):
    #     print(f"Document {i}:\n{doc.page_content}\n")
    #     if doc.metadata:
    #         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    return relevant_doc
query="What did elon musk told his employees?"

documents=query_vectorstore(query)

model=ChatOpenAI(model='gpt-4o')

chat_template=ChatPromptTemplate.from_messages([
    ("system","You are an helpful assistant you are required to answer user's question  based on the data given. User has some document/text/data he only what his answer to be generated based on the given data. If answer is not given in data reply with 'Answer is not available in provided data here's what most relevant is:`give user of and idea of what he can ask`' and if question is irrelevant reply with 'Please stick to the topic `Provide topic desciption here`' \n\n Data: "+"\n\n".join([doc.page_content for doc in documents])),
    ("human","{query}")
])

prompt=chat_template.invoke({'query':query})
result=model.invoke(prompt)

print(result.content)