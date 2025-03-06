
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI


load_dotenv()

model=ChatOpenAI()

prompt_template= ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert product reviewer")
        ,("human","List main feature of this product : {product}")
    ]
)

def pros_template(features):
    prostemp=ChatPromptTemplate.from_messages([
        ("system","You are an expert product reviewer."),
        ("human","based on these features {features} list out only pros, {features}")
    ])
    
    return prostemp.format_prompt(features=features)

def cons_template(features):
    prostemp=ChatPromptTemplate.from_messages([
        ("system","You are an expert product reviewer."),
        ("human","based on these features list out only cons, features: {features}")
    ])
    
    return prostemp.format_prompt(features=features)


def combine_pros_cons(pros,cons):
    return f"Pros : {pros}, Cons: {cons}"

pros_branch_chain = (RunnableLambda(lambda x: pros_template(x))|model| StrOutputParser())
cons_branch_chain=(RunnableLambda(lambda x: cons_template(x))| model| StrOutputParser())

chain=(prompt_template
       | model 
       | StrOutputParser()
       | RunnableParallel(branches={"pros":pros_branch_chain,"cons":cons_branch_chain})
       | RunnableLambda(lambda x:  combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"])))

result=chain.invoke({'product':'iPhone X'})

print(result)