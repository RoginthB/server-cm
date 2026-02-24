from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.graph import END, START,StateGraph,MessagesState
import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import List,Optional,Dict,Any
from langchain.agents.structured_output import ToolStrategy
load_dotenv()

llm = init_chat_model("google_genai:gemini-2.5-flash")

def get_web_content(state):
    search = GoogleSerperAPIWrapper()
    searched_content =[]
    for query in state.rewrited_query:
        searched_content.append(search.run(query))
    state.web_content=searched_content
    return state

def rewrite_query(state):
    SYSTEM_PROMPT =f"""
        Your a helpful aisstant to Rewrite User querie into Search optimized queries
        -expand abbreviations 
        -add context keywords
        -split multi-intent queries
        Use the Following details also to rewrite the user querie
        {state.query_intent}
    """
    agent = create_agent(llm ,system_prompt=SYSTEM_PROMPT,response_format=ToolStrategy(Query_list))
    response =agent.invoke({"messages": [{"role": "user", "content": state.query}]})
    state.rewrited_query = response["structured_response"].rewrited_query
    return state

def understand_query(state):
    SYSTEM_PROMPT ="""
You are a Query Understanding module inside a production web search agent.
Your job is to analyze the user query and decide:
1. The primary intent of the query
2. Whether the query requires up-to-date or real-time information
3. Whether a web search is required
4. Any important constraints or entities mentioned by the user

You must NOT answer the user question.
You must NOT hallucinate information.
You must ONLY analyze the query.
Classify intent into ONE of the following:
- factual
- how_to
- comparison
- troubleshooting
- news
- opinion
- navigational
- ambiguous

Freshness rules:
- Freshness is REQUIRED if the query mentions:
  "latest", "today", "current", "now", specific years, recent events, prices, releases, news
- Freshness is NOT required for general knowledge, definitions, or timeless concepts

Web search rules:
- Web search is REQUIRED if freshness is required
- Web search is REQUIRED if the answer depends on external or changing information
- Web search is NOT required for well-known static knowledge

Extract important entities such as:

- Technologies
- Product names
- Versions
- Locations
- Dates
- Organizations

If the query is ambiguous or missing context, mark intent as "ambiguous" and explain why.
    """
    understand_query_agent = create_agent(llm, system_prompt=SYSTEM_PROMPT,response_format=ToolStrategy(Intent))
    result = understand_query_agent.invoke({"messages": [{"role": "user", "content": state.query}]})
    state.query_intent.append(result["structured_response"].model_dump())
    return state
   
def route_node(state):
    if state.query_intent[0]["requires_web_search"]:
        return "rewrite_query"
    else:
        return "answer_sythesis"

def filter_results(state):
    pass

def fetch_content(state):
    pass

def answer_sythesis(state):
    SYSTEM_PROMPT=f"""

[ROLE]
You are a helpful, concise assistant that answers user questions clearly, politely, and step-by-step when needed. 
Always prefer accuracy, brevity, and actionable guidance.

[OBJECTIVE]
Given the user's message, produce a direct answer in a friendly tone, with short paragraphs and lists where helpful.
If the request is ambiguous, ask one clarifying question (max 1).
If the request needs external or recent info you don't have, say so and suggest what info is needed.
You should answer based on the CONTEXT.

[STYLE]
- Be courteous, professional, and approachable.
- Use simple language; avoid jargon unless the user is technical.
- Use headings and bullet points when beneficial.
- Keep answers tight; avoid fluff.
- If giving steps, number them.
- If you need to show code, wrap it in fenced code blocks.

[SAFETY & BOUNDS]
- Do not fabricate facts. If unsure, be explicit about uncertainty.
- Do not provide disallowed or harmful content.
- If the user asks for illegal, unsafe, or sexual content, refuse and suggest a safer alternative.

[CONTEXT]
{state}

[NOTES]
- If the user asks for code, provide minimal runnable examples.

    """
    answer_agent = create_agent(llm,system_prompt=SYSTEM_PROMPT)
    result=answer_agent.invoke({"messages": [{"role": "user", "content": state.query}]})
    state.response = result["messages"][-1].content
    return state

class WebSearchGraph(BaseModel):
    query:str
    query_intent: List[Dict[str, Any]] = Field(default_factory=list)
    rewrited_query:List[str] =None
    web_content:List[str]=None
    response:str=None
    
    
class Query_list(BaseModel):
    rewrited_query:List[str]

class Intent(BaseModel):
    intent: str
    requires_freshness:bool
    requires_web_search: bool
    entities: List[str]
    constraints:List[str]
    reasoning: str


builder = StateGraph(WebSearchGraph)
builder.add_node("understand_query",understand_query)
builder.add_node("rewrite_query",rewrite_query)
builder.add_node("get_web_content",get_web_content)
builder.add_node("answer_sythesis",answer_sythesis)

builder.add_edge(START,"understand_query")
builder.add_conditional_edges("understand_query",
                              route_node,
                              {"rewrite_query":"rewrite_query",
                               "answer_sythesis":"answer_sythesis"
                               }
                              )
builder.add_edge("rewrite_query","get_web_content")
builder.add_edge("get_web_content","answer_sythesis")
builder.add_edge("answer_sythesis",END)

graph =builder.compile()


