from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from typing import Literal
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage


load_dotenv()

tools = [TavilySearchResults(max_results=1)]

tool_node = ToolNode(tools)

model = AzureChatOpenAI(
    api_version="2023-12-01-preview",
    api_key="22104a245e834cb0b49abc9009dfe4c9",
    azure_deployment="langgraph2",
    azure_endpoint="https://langraphwork.openai.azure.com/",
)

model = model.bind_tools(tools)


def add_messages(left: list, right: list):
    """Add-don't-overwrite."""
    return left + right


class AgentState(TypedDict):
    # The `add_messages` function within the annotation defines
    # *how* updates should be merged into the state.
    messages: Annotated[list, add_messages]


# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return "__end__"


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_node("tools", tool_node)
workflow.add_edge("tools", "agent")

workflow.set_entry_point("agent")


app = workflow.compile()

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for output in app.stream(inputs, stream_mode="updates"):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")
