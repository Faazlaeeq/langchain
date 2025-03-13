import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
current_dir=os.path.dirname(os.path.abspath(__file__))
presistent_dir=os.path.join(current_dir, "db",'chroma_db')

embeddings=OpenAIEmbeddings(model='text-embedding-3-small')

db=Chroma(persist_directory=presistent_dir,embedding_function=embeddings)

query="Who is odysseus' wife?"

retriver= db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},

)

relevant_docs=retriver.invoke(query)

print("\n Relevant Documents: \n\n")

for i,doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
