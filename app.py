import os
from dotenv import load_dotenv

# Load environment variables FIRST before importing the engine
load_dotenv()
# FORCE REST transport to prevent gRPC thread-auth drops in Flask workers
os.environ["GOOGLE_API_TRANSPORT"] = "rest"

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_core.messages import HumanMessage

# Import the engine and retry helper we created earlier
from reasoning_engine import app as agent_app, execute_with_retry

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')
    thread_id = data.get('thread_id', 'web_session_1')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
        
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # invoke the RAG planner agent
        final_state = execute_with_retry(
            agent_app.invoke,
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        bot_response = final_state["messages"][-1].content
        
        # Handle cases where the LLM returns a list of content blocks instead of a string
        if isinstance(bot_response, list):
            texts = []
            for block in bot_response:
                if isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
                elif isinstance(block, str):
                    texts.append(block)
            bot_response = "\n".join(texts)
            
        return jsonify({
            'response': bot_response,
            'thread_id': thread_id
        })
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask server with debug=False and threaded=False 
    # This prevents gRPC multi-threading authentication drops on Windows!
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=False)