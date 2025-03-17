import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage,SystemMessage


load_dotenv()

current_dir= os.path.dirname(os.path.abspath(__file__))
persistant_dir=os.path.join(current_dir,'db','oneoff_db')


embedder=OpenAIEmbeddings(model="text-embedding-3-small")

db=Chroma(persist_directory=persistant_dir,embedding_function=embedder)

retriever=db.as_retriever(search_type='similarity',search_kwargs={'k':1})

llm=ChatOpenAI(model='gpt-4o')

contextual_retriver_query= (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

crq_template=ChatPromptTemplate.from_messages([
    ("system",contextual_retriver_query),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])

history_aware_retriver=create_history_aware_retriever(llm,retriever,crq_template)

qa_system_query=(   "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}")

qa_template=ChatPromptTemplate.from_messages([
    ('system',qa_system_query),
    MessagesPlaceholder("chat_history"),
    ('human',"{input}")
])

qa_chain=create_stuff_documents_chain(llm,qa_template)

rag_chain=create_retrieval_chain(history_aware_retriver,qa_chain)


def continue_chat():
    chat_history=[]
    while True:
        user_input=input("You: ")
        if user_input.lower() =="exit":
            break
        result=rag_chain.invoke({"input":user_input,"chat_history":chat_history})
        
        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(SystemMessage(content=result['answer']))


# Main function to start the continual chat
if __name__ == "__main__":
    continue_chat()