from __future__ import annotations

import operator
from typing import TypedDict, List, Annotated
from pathlib import Path
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="What to cover")

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str

llm = ChatNVIDIA(
    model = "nvidia/nemotron-3.5-lightning-30b-a3b",
    model_kwargs={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)

def orchestrator(state : State):

    plan = llm.with_structured_output(Plan,method ="json_mode").invoke(
        [
            ("system", "Create a blog plan with 3 sections on the following topic."),
            ("user",f"Topic = {state['topic']}")
        ]
    )
    return {
        "plan" : plan
    }

def fanout(state : State):
    task_list = []
    for task in state['plan'].tasks:
        task_list.append(Send("worker",{"task": task, "topic": state["topic"], "plan": state["plan"]}))
    return task_list


def worker(payload: dict) -> dict:

    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    blog_title = plan.blog_title

    section_md = llm.invoke(
        [
            ("system" ,"Write one clean Markdown section."),
            (       "user",
                    f"""Blog: {blog_title}\n"
                    Topic: {topic}\n\n"
                    Section: {task.title}\n"
                    Brief: {task.brief}\n\n"
                    "Return only the section content in Markdown."""
            )
        ]
    ).content.strip()

    return {"sections": [section_md]}

def reducer(state: State) -> dict:
    
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    filename = title.lower().replace(" ", "_") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return {"final": final_md}


g = StateGraph(State)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)


DATABASE_URL = os.getenv("DATABASE_URL")
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()
app = g.compile(checkpointer=checkpointer)
config = {
        "configurable": {
            "thread_id": "test_thread_id"
        }
    }
out = app.invoke({"topic": "Write a blog on Self Attention", "sections": []}, config=config)
print(out)




