from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="", 
)

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French. Translate the user sentence."),
    HumanMessage(content="I love programming."),
]

response = llm.invoke(messages)


print(response.content)
