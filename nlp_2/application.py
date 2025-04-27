import langchain
import os
from random import randint
from flask import Flask, render_template, request, session, render_template_string
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

application = Flask(__name__)
application.secret_key = os.getenv("FLASK_SECRET_KEY")

LANGCHAIN_TRACING_V2 = True

# Load environment variables from a .env file
load_dotenv()

# Store environment variables in separate variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
index_name = os.getenv("PINECONE_BARRIA_INDEX_NAME")
namespace = os.getenv("PINECONE_NAMESPACE")


from langchain_groq import ChatGroq

chat = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",  # Or "llama3-8b-8192", etc.
    temperature=0,
    streaming=True
)

### EMBEDDINGS
from utils import SentenceTransformerWrapper

# Load the sentence transformer model
raw_model = SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)
# Wrap it for LangChain compatibility
embed_model = SentenceTransformerWrapper(raw_model)
print(len(embed_model.embed_query('hola')))

# CONNECT TO PINECONE DATABASE

pc=Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore(
    pinecone_api_key = PINECONE_API_KEY,
    index_name=index_name,
    embedding=embed_model,
    namespace=namespace,
)
retriever=vectorstore.as_retriever()


# Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant specialized in answering questions based on provided context. "
    "Carefully analyze the context below, derived from a CV and LinkedIn profile, to accurately respond. "
    "If the answer is not explicitly found in the context, state clearly that you don't have enough information. "
    "Be concise and professional."
    "\n\n"
    "Take into account that all the context provided is related to Alex Barria."
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chat, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    print("session_id", session_id)
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def get_completion(usr_txt, session_id):
    conversational_rag_chain.invoke(
        {"input": usr_txt},
        config={
            "configurable": {"session_id": session_id}
        },  # constructs a key "abc123" in `store`.
    )["answer"]
    last_message = store[session_id].messages[-1].content
    return last_message

@application.route("/")

def home():
    if 'session_id' not in session:
        session['session_id'] = randint(0, 9999) 
    return render_template("./index2.html")
@application.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')  
    session_id = session.get('session_id')
    response = get_completion(userText, session_id) 
     #return str(bot.get_response(userText)) 
    return response

if __name__ == "__main__":
    application.debug = True
    application.run()
