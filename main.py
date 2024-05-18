from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import END, MessageGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import List, Literal


# Define tools for multiplication and averaging
@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number


@tool
def average(first_number: int, second_number: int):
    """Calculates the average of two numbers."""
    return (first_number + second_number) / 2


# Router function to direct flow based on tool calls
def router(state: List[BaseMessage]) -> Literal["multiply", "__end__", "average"]:
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    if any(call["function"]["name"] == "average" for call in tool_calls):
        return "average"
    elif any(call["function"]["name"] == "multiply" for call in tool_calls):
        return "multiply"
    else:
        return "__end__"


# Initialize the model
model = AzureChatOpenAI(
    api_version="2023-12-01-preview",
    api_key="22104a245e834cb0b49abc9009dfe4c9",
    azure_deployment="langgraph2",
    azure_endpoint="https://langraphwork.openai.azure.com/",
)

# Bind tools to the model
model_with_tools = model.bind_tools([multiply, average])

# Build the message graph
builder = MessageGraph()

builder.add_node("oracle", model_with_tools)
builder.add_conditional_edges("oracle", router)

# Add tool nodes and edges
builder.add_node("multiply", ToolNode([multiply]))
builder.add_node("average", ToolNode([average]))

builder.add_edge("average", END)
builder.add_edge("multiply", END)

# Set the entry point and compile the graph
builder.set_entry_point("oracle")

runnable = builder.compile()

# Invocation example
print(runnable.invoke(HumanMessage("What is 123 * 456?")))
print(runnable.invoke(HumanMessage("What is the average of 10 and 20?")))
