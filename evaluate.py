import os
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Import our compiled graph from reasoning_engine.py
from reasoning_engine import app
# Load environment variables
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set.")

print("Initializing Evaluation Pipeline...")
eval_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

# Define structured output for the Judge
class EvaluationScore(BaseModel):
    score: int = Field(description="Score from 1 to 5.", ge=1, le=5)
    reasoning: str = Field(description="Explanation for the score.")

judge = eval_llm.with_structured_output(EvaluationScore)

def execute_with_retry(func, *args, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                print(f"    [RATE LIMIT] Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    return None

# ==========================================
# 1. THE GOLDEN DATASET (Ground Truth)
# ==========================================
# Hand-crafted questions and perfect answers based on the OmniTech document
golden_dataset = [
    {
        "question": "Did OmniTech acquire CyberDyne in Q3?",
        "ground_truth": "No, OmniTech withdrew from acquisition discussions due to valuation misalignment, integration risk, and unresolved intellectual property considerations."
    },
    {
        "question": "What was the Q2 Revenue for Cloud Compute, and was it impacted by anything specific?",
        "ground_truth": "The Q2 Revenue for Cloud Compute was $138.7 million. It was impacted by a one-time tax rebate of $4.2 million."
    },
    {
        "question": "Why did the European sector revenue plummet by 22%?",
        "ground_truth": "European sector revenue plummeted by 22% due to regulatory fines, contract suspensions, and delayed renewals linked to data localization mandates."
    }
]

# ==========================================
# 2. RUN EVALUATION
# ==========================================
print(f"Starting Evaluation on {len(golden_dataset)} test cases...\n")

total_score = 0

for i, test_case in enumerate(golden_dataset):
    print(f"--- Test Case {i+1} ---")
    print(f"Q: {test_case['question']}")
    
    # 1. Generate Answer using our RAG Engine
    config = {"configurable": {"thread_id": f"eval_session_{i}"}}
    try:
        final_state = execute_with_retry(app.invoke, {"messages": [HumanMessage(content=test_case['question'])]}, config=config)
        generated_answer = final_state["messages"][-1].content
    except Exception as e:
        print(f"[ERROR] Engine failed to generate answer: {e}")
        continue
        
    print(f"\nGenerated Answer: {generated_answer}")
    print(f"Ground Truth:     {test_case['ground_truth']}")
    
    # 2. Evaluate using LLM-as-a-Judge
    eval_prompt = (
        f"You are an expert evaluator grading an AI assistant's answer against a Ground Truth.\n"
        f"Score the Generated Answer from 1 to 5 based purely on its factual accuracy matching the Ground Truth.\n"
        f"1 = Completely wrong or missing.\n"
        f"3 = Partially correct.\n"
        f"5 = Perfectly captures the core facts of the Ground Truth.\n\n"
        f"Ground Truth: {test_case['ground_truth']}\n"
        f"Generated Answer: {generated_answer}"
    )
    
    result = execute_with_retry(judge.invoke, eval_prompt)
    if result:
        print(f"\n>> Judge Score: {result.score}/5")
        print(f">> Reason:      {result.reasoning}\n")
        total_score += result.score
        
    # Sleep to avoid rate limits
    time.sleep(2)

print("="*40)
print(f"FINAL EVALUATION SCORE: {total_score}/{len(golden_dataset) * 5}")
print("="*40)
