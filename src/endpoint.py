import json
from http import HTTPStatus
from starlette.responses import Response
from fastapi import APIRouter
from src.agent import graph
from pydantic import BaseModel

router = APIRouter()


class AgentRequest(BaseModel):
    query:str

class AgentResponse(BaseModel):
    response:dict

@router.get("/")
def root():
    return {"message":"hello"}

@router.post("/")
async def run_agent(query:AgentRequest)->Response:
    res = await graph.ainvoke(query)
    return Response(
        content=json.dumps({"result":res["response"]}),
        status_code= HTTPStatus.ACCEPTED,
    )

