from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.graph import END, START,StateGraph,MessagesState
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

def get_web_content(state):
    search = GoogleSerperAPIWrapper()
    state.web_content=search.run(state.query)

def rewrite_query(state):

    return state

class WebSearchGraph(StateGraph):
    query:str

graph = StateGraph(WebSearchGraph)
graph.add_node("web_search",get_web_content)

