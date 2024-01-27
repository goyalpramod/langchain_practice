import os 
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]  


llm = ChatOpenAI(openai_api_key = openai_api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print(chain.invoke({"input": "how can langsmith help with testing?"}))