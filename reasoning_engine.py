import os
import pickle
import time
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb

# Load environment variables
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set.")

print("Loading LLMs and Embeddings...")
api_key = os.environ.get("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, api_key=api_key, transport="rest")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key, transport="rest")

print("Loading ChromaDB Vector Store...")
db_dir = "./chroma_db"
client = chromadb.PersistentClient(path=db_dir)
vector_store = Chroma(
    client=client,
    collection_name="sec_10k_collection",
    embedding_function=embeddings,
)

print("Loading local BM25 Keyword Search Index...")
with open("bm25_index.pkl", "rb") as f:
    bm25_retriever = pickle.load(f)

# ==========================================
# 1. DEFINE AGENT TOOLS
# ==========================================

@tool
def semantic_search(query: str) -> str:
    """Useful for answering conceptual questions, finding underlying meaning, or when searching for broader topics.
    Input should be a fully formed sentence or question."""
    results = vector_store.similarity_search(query, k=3)
    
    if not results:
        return "No relevant documents found via semantic search."
        
    formatted_results = []
    for doc in results:
        formatted_results.append(f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'Unknown')}")
        
    return "\n\n---\n\n".join(formatted_results)

@tool
def keyword_search(query: str) -> str:
    """Useful for finding exact names, numbers, unique identifiers, specific jargon, or strict exact wording.
    Input should be 1-3 specific keywords, NOT a full sentence."""
    results = bm25_retriever.invoke(query)
    
    if not results:
        return "No exact keyword matches found."
        
    formatted_results = []
    # Limit to top 2 to preserve context window
    for doc in results[:2]:
        formatted_results.append(f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'Unknown')}")
        
    return "\n\n---\n\n".join(formatted_results)

tools = [semantic_search, keyword_search]
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 2. DEFINE THE REASONING ENGINE (LANGGRAPH)
# ==========================================
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from pydantic import BaseModel, Field

# ==========================================
# 2. DEFINE EVALUATORS (Self-Correction)
# ==========================================

class GroundedEvaluation(BaseModel):
    is_grounded: bool = Field(description="True if the answer is strictly based on the retrieved context, False if it hallucinates.")
    critique: str = Field(description="If not grounded, explain what is hallucinated to help the agent correct it.")

class RelevanceEvaluation(BaseModel):
    is_relevant: bool = Field(description="True if the answer directly addresses the user's question, False otherwise.")
    critique: str = Field(description="If not relevant, explain what was missed to help the agent correct it.")

evaluator_llm = llm.with_structured_output(GroundedEvaluation)
relevance_llm = llm.with_structured_output(RelevanceEvaluation)

# Define the Graph State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def execute_with_retry(func, *args, **kwargs):
    """Helper to wrap LLM calls with rate limit handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait_time = 60 * (attempt + 1)
                print(f"    [RATE LIMIT] Quota exceeded on attempt {attempt+1}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e # re-raise if it's not a rate limit issue
    raise Exception("Max retries exceeded for API call.")

def planner_agent(state: AgentState):
    """The central brain that decides whether to search, use tools, or answer."""
    print("\n[PLANNER] Agent is thinking...")
    sys_msg = SystemMessage(content="You are a brilliant financial analyst reasoning engine. "
                                   "You have access to a semantic search tool and an exact keyword search tool. "
                                   "Always use the tools to answer questions about OmniTech or the SEC documents. "
                                   "If a user asks a complex question, you can use the tools multiple times.")
    
    messages = [sys_msg] + state["messages"]
    response = execute_with_retry(llm_with_tools.invoke, messages)
    return {"messages": [response]}

def tool_executor(state: AgentState):
    """Executes the specific search tool the Planner requested."""
    messages = state["messages"]
    # Get the last message which should be the AI requesting a tool call
    last_message = messages[-1]
    
    tool_responses = []
    for tool_call in last_message.tool_calls:
        print(f"[TOOL EXECUTOR] Running: {tool_call['name']} with args {tool_call['args']}")
        
        # Dispatch to correct function
        if tool_call["name"] == "semantic_search":
            result = semantic_search.invoke(tool_call["args"])
        elif tool_call["name"] == "keyword_search":
            result = keyword_search.invoke(tool_call["args"])
        else:
            result = "Error: Tool not found."
            
        # Append the tool response back as a ToolMessage
        tool_responses.append(ToolMessage(
            content=str(result), 
            tool_call_id=tool_call["id"],
            name=tool_call["name"]
        ))
        
    return {"messages": tool_responses}

def should_continue(state: AgentState):
    """Router to dictate if we keep going or stop."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM didn't request any tools, it means it's ready to present the final answer
    if not last_message.tool_calls:
        return "audit"
    
    # Otherwise, it requested a tool, so route to the tool executor
    return "tools"

def auditor_node(state: AgentState):
    """Checks if the LLM's final answer is grounded in the tool context (No Hallucinations)."""
    print("\n[AUDITOR] Checking for hallucinations...")
    messages = state["messages"]
    
    # Extract the user's latest question and the drafted answer
    original_question = next(msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage))
    draft_answer = messages[-1].content
    
    # Extract all retrieved context from ToolMessages
    context = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            context.append(msg.content)
            
    context_str = "\n".join(context)
    
    if not context_str.strip():
        # If no tools were used, we can't really grounding-check against context, skip to relevance
        print("  -> No context retrieved, skipping grounding check.")
        return {"messages": []}

    eval_prompt = (
        f"You are a strict Auditor. Your job is to check if the 'Draft Answer' is strictly based ONLY on the 'Context'. "
        f"If the answer contains claims not present in the context, it is hallucinated (False).\n\n"
        f"Context:\n{context_str}\n\nDraft Answer: {draft_answer}"
    )
    
    result = execute_with_retry(evaluator_llm.invoke, eval_prompt)
    
    if not result.is_grounded:
        print(f"  -> HALLUCINATION DETECTED: {result.critique}")
        # Send a system message back to the planner to fix it
        return {"messages": [SystemMessage(content=f"Your previous answer failed the grounding check. Auditor Feedback: {result.critique}. Please revise your answer to only include facts from the search context, or use tools to find the missing information.")]}
    
    print("  -> Passed. Answer is grounded in context.")
    return {"messages": []}

def gatekeeper_node(state: AgentState):
    """Checks if the drafted answer actually addresses the user's question."""
    print("\n[GATEKEEPER] Checking for relevance...")
    messages = state["messages"]
    
    # Is the last message from the Auditor telling it to try again? Skip relevance check.
    if isinstance(messages[-1], SystemMessage) and "Auditor Feedback" in messages[-1].content:
        return {"messages": []}
        
    # Get the drafted answer (might be the last or second to last depending on Auditor)
    draft_answer = next(msg.content for msg in reversed(messages) if isinstance(msg, AIMessage) and not msg.tool_calls)
    original_question = next(msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage))
    
    eval_prompt = (
        f"You are a Gatekeeper. Does the 'Draft Answer' directly answer the user's 'Question'?\n\n"
        f"Question: {original_question}\n\nDraft Answer: {draft_answer}"
    )
    
    result = execute_with_retry(relevance_llm.invoke, eval_prompt)
    
    if not result.is_relevant:
        print(f"  -> NOT RELEVANT: {result.critique}")
        return {"messages": [SystemMessage(content=f"Your previous answer was not quite relevant. Gatekeeper Feedback: {result.critique}. Please revise your answer or search for better information.")]}
        
    print("  -> Passed. Answer is relevant.")
    return {"messages": []}

def check_auditor_gatekeeper_results(state: AgentState):
    """Determines if the Planner needs to re-run based on Auditor or Gatekeeper feedback."""
    last_message = state["messages"][-1]
    
    # If the last message is a SystemMessage containing feedback, loop back to the planner
    if isinstance(last_message, SystemMessage) and ("Feedback" in last_message.content):
        return "planner"
        
    return "end"

print("Building LangGraph Multi-Agent Engine...")
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("planner", planner_agent)
workflow.add_node("tools", tool_executor)
workflow.add_node("auditor", auditor_node)
workflow.add_node("gatekeeper", gatekeeper_node)

# Add Edges
workflow.set_entry_point("planner")

# After planner, we hit the router to see if we go to tools or the auditor
workflow.add_conditional_edges(
    "planner",
    should_continue,
    {
        "tools": "tools",
        "audit": "auditor"
    }
)

# Tools always return their findings back to the planner
workflow.add_edge("tools", "planner")

# Auditor flows into Gatekeeper
workflow.add_edge("auditor", "gatekeeper")

# Gatekeeper routes to END (if passed) or back to Planner (if failed)
workflow.add_conditional_edges(
    "gatekeeper",
    check_auditor_gatekeeper_results,
    {
        "planner": "planner",
        "end": END
    }
)

# Compile the graph
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("\n\n" + "="*50)
    print("OmniTech Reasoning Engine Online (With Memory)")
    print("Type 'exit' or 'quit' to quit.")
    print("="*50 + "\n")
    
    # We use a static thread_id for this session so it remembers the entire chat
    config = {"configurable": {"thread_id": "session_1"}}
    
    while True:
        user_input = input("User >> ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        print("\n--- Reasoning Trace ---")
        # Run the graph
        final_state = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        
        print("\n--- Final Answer ---")
        print(final_state["messages"][-1].content)
        print("\n")