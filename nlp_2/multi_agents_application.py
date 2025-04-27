import os
from random import randint
from flask import Flask, render_template, request, session
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

import os
from random import randint
from flask import Flask, render_template, request, session
from dotenv import load_dotenv
from agent_test import Agent

# Flask app
application = Flask(__name__)
application.secret_key = os.getenv("FLASK_SECRET_KEY")

# Load environment variables
load_dotenv()

# Store environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
namespace = os.getenv("PINECONE_NAMESPACE")

# ========== Set up Agent ========== #

from nlp_2.config import agent_prompts

agent = Agent(model="groq", agent_prompts=agent_prompts)

# Session storage
store = {}

def get_session_state(session_id: str) -> dict:
    if session_id not in store:
        store[session_id] = {"messages": []}
    return store[session_id]

def update_session_state(session_id: str, new_state: dict):
    store[session_id] = {"messages": new_state["messages"]}  # Solo guardamos los mensajes

def get_completion(user_input, session_id):
    state = get_session_state(session_id)

    input_state = {"messages": [HumanMessage(content=user_input)]}

    new_state = agent.graph.invoke(input_state)

    updated_messages = state["messages"] + [HumanMessage(content=user_input)] + new_state["messages"]
    update_session_state(session_id, {"messages": updated_messages})

    last_message = new_state['messages'][-1].content
    return last_message

# ========== Routes ========== #

@application.route("/")
def home():
    if 'session_id' not in session:
        session['session_id'] = str(randint(0, 9999))
    return render_template("./index_multi_agents.html")

@application.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')  
    session_id = session.get('session_id')
    response = get_completion(userText, session_id) 
    return response

# ========== Main ========== #

if __name__ == "__main__":
    application.debug = True
    application.run()