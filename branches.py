from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI()


positive_template= ChatPromptTemplate.from_messages([
    ('system','your are a helpful assistant'),
    ('human','Generate a thank you message for this positive feedback : {feedback}')
])

negative_template =ChatPromptTemplate.from_messages([
        ('system','your are a helpful assistant'),
        ('human',         "Generate a response addressing this negative feedback: {feedback}."),

])

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

classification_template = ChatPromptTemplate.from_messages([
    ('system','You are being used as a feedback nature detector in 4 categories [positive,negative,neutral,esclate] answer in one word catergory'),
    ('human','what is nature of this feedback: {feedback}')
])

branches = RunnableBranch((
    lambda x: 'positive' in x,
    positive_template | model | StrOutputParser()
),
(lambda x: 'negative' in x,
                           negative_template| model | StrOutputParser),
(lambda x: 'neutral' in x,
 neutral_feedback_template | model | StrOutputParser()),
escalate_feedback_template|model | StrOutputParser())

classification_chain= classification_template|model | StrOutputParser()

chain= classification_chain | branches

review="The product is okay. It works as expected but nothing exceptional."
result= chain.invoke({'feedback':review})

print(result)