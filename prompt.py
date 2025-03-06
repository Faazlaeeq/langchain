from langchain.prompts import ChatPromptTemplate
messages=[
    ("system","You are a comedian who tell jokes about {type}")
    ,("human","Tell me a joke about {topic}")
]

prompt_template= ChatPromptTemplate.from_messages(messages)
prompt=prompt_template.invoke({"type":"animal","topic":"cat"})
print(prompt)
