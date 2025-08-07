from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize the LLM with the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY
)

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French. Translate the user sentence."),
    HumanMessage(content="I love programming."),
]

while True:
    user_input = input("Enter a sentence to translate (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    messages[1] = HumanMessage(content=user_input)
    response = llm.invoke(messages)
    print("Translation:", response.content)
