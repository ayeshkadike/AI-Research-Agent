import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.schema import SystemMessage
from pydantic import BaseModel, Field
from typing import Type

# Load environment variables
load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# Search tool using the Serper API
def search(query):
    url = "https://google.serper.dev/search"
    headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps({"q": query}))
    print(response.text)
    return response.text

# Scrape and summarize website content
def scrape_website(objective, url):
    print("Scraping website...")
    headers = {'Cache-Control': 'no-cache', 'Content-Type': 'application/json'}
    data_json = json.dumps({"url": url})
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)
        return text if len(text) <= 10000 else summarize_content(objective, text)
    else:
        print(f"HTTP request failed with status code {response.status_code}")
        return ""

# Summarize content based on the objective
def summarize_content(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    prompt_template = PromptTemplate(template="Write a summary for {objective}: '{text}'\nSUMMARY:", input_variables=["text", "objective"])
    summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', map_prompt=prompt_template, combine_prompt=prompt_template, verbose=True)
    return summary_chain.run(input_documents=docs, objective=objective)

# ScrapeWebsiteTool definition
class ScrapeWebsiteInput(BaseModel):
    objective: str = Field(description="Objective of the user")
    url: str = Field(description="URL of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "Scrapes and summarizes website content."
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

# Initialize LangChain agent with tools
tools = [
    Tool(name="Search", func=search, description="Search tool for current events and data."),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(content="Guidelines for the world class researcher agent.")

agent_kwargs = {"extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")], "system_message": system_message}
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, agent_kwargs=agent_kwargs, memory=memory)

# Streamlit web application
def main():
    import streamlit as st
    st.set_page_config(page_title="AI Research Agent", page_icon=":bird:")
    st.header("AI Research Agent :bird:")
    query = st.text_input("Research Goal")

    if query:
        st.write("Researching:", query)
        result = agent({"input": query})
        st.info(result['output'])

if __name__ == '__main__':
    main()
