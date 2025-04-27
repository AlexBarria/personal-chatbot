# %%
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from sentence_transformers import SentenceTransformer
from nlp_2.utils import SentenceTransformerWrapper
import os
import regex as re
from nlp_2.config import agent_prompts
# %%
from typing import TypedDict, Annotated
import operator
import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain.chains import RetrievalQA 

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
# %%
from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
namespace = os.environ.get('PINECONE_NAMESPACE')
barria_index_name = os.environ.get('PINECONE_BARRIA_INDEX_NAME')
bureu_index_name = os.environ.get('PINECONE_BUREU_INDEX_NAME')

# %%
class Agent:

    def __init__(self, model, agent_prompts=""):
        self.system = agent_prompts["system_prompt"]
        self.alex_bot_prompt = agent_prompts["alex_bot_prompt"]
        self.clara_bot_prompt = agent_prompts["clara_bot_prompt"]
        self.llm_model = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            streaming=True
        )
        self._sentence_transformer_raw_model = SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)
        self._sentence_transformer = SentenceTransformerWrapper(self._sentence_transformer_raw_model)
        self._vectorstore_cv_barria = PineconeVectorStore(
            index_name=barria_index_name,
            embedding=self._sentence_transformer,
            namespace=namespace,
        )
        self._vectorstore_cv_bureu = PineconeVectorStore(
            index_name=bureu_index_name,
            embedding=self._sentence_transformer,
            namespace=namespace,
        )

        # Mapping names to agent functions
        self.name_to_agent = {
            "Alex": self.call_openai_alex,
            "Clara": self.call_openai_clara,
        }

        self._qa_barria_cv = RetrievalQA.from_chain_type(
            llm=self.llm_model,
            chain_type="stuff",
            retriever=self._vectorstore_cv_barria.as_retriever()
        )

        self._qa_bureu_cv = RetrievalQA.from_chain_type(
            llm=self.llm_model,
            chain_type="stuff",
            retriever=self._vectorstore_cv_bureu.as_retriever()
        )

        graph = StateGraph(AgentState)

        # Nodes
        graph.add_node("regex", self.pass_through)  # Just routes to dispatch
        graph.add_node("dispatch_agents", self.dispatch_agents)
        graph.add_node("concatenate_responses", self.concatenate_responses)
        graph.add_node("llm", self.call_openai_llm)

        # Edges
        graph.add_edge("regex", "dispatch_agents")
        graph.add_edge("dispatch_agents", "concatenate_responses")
        graph.add_edge("concatenate_responses", "llm")

        graph.set_entry_point("regex")
        self.graph = graph.compile()

    def pass_through(self, state: AgentState):
        # Acts as a dummy node for routing
        return state

    def dispatch_agents(self, state: AgentState):
        content = state['messages'][-1].content
        responses = []

        for name, agent_func in self.name_to_agent.items():
            if re.search(rf"\b{name}\b", content, re.IGNORECASE):
                response = agent_func(state)['messages'][0]
                responses.append(response)

        return {'messages': responses}

    def concatenate_responses(self, state: AgentState):
        responses = [msg.content for msg in state['messages'] if isinstance(msg, ToolMessage)]
        # print(responses)
        concatenated = "\n".join(responses)
        return {'messages': [ToolMessage(content=concatenated, tool_call_id="concat")]}

    def call_openai_alex(self, state: AgentState):
        question = state['messages'][-1].content
        query = f"{self.alex_bot_prompt} {question}"
        llm_response = self._qa_barria_cv.invoke(query)
        message_to_final_agent = "Alex's bot response: " + llm_response["result"]
        # print(message_to_final_agent)
        return {'messages': [ToolMessage(content=message_to_final_agent, tool_call_id="alex_bot")]}

    def call_openai_clara(self, state: AgentState):
        question = state['messages'][-1].content
        query = f"{self.clara_bot_prompt} {question}"
        llm_response = self._qa_bureu_cv.invoke(query)
        message_to_final_agent = "Clara's bot response: " + llm_response["result"]
        # print(message_to_final_agent)
        return {'messages': [ToolMessage(content=message_to_final_agent, tool_call_id="clara_bot")]}

    def call_openai_llm(self, state: AgentState):
        # Recupera los mensajes de entrada
        messages = state['messages']

        # Extrae el contenido de cada ToolMessage
        previous_responses = [msg.content for msg in messages if isinstance(msg, ToolMessage)]

        # Une todos los mensajes previos en un solo texto
        prompt = "\n".join(previous_responses)

        # Prepara los mensajes para el LLM, agregando el system prompt si está definido
        final_messages = []
        if self.system:
            final_messages.append(SystemMessage(content=self.system))
        final_messages.append(HumanMessage(content=prompt))

        # Llama al modelo de lenguaje real usando ChatGroq (ya inicializado como self.llm_model)
        print("Final messages to LLM:", final_messages)
        final_response = self.llm_model.invoke(final_messages)
        print("Final response from LLM:", final_response)
        return {'messages': [AIMessage(content=final_response.content)]}

# %%
# ChatOpenAI.api_key = os.getenv("OPENAI_API_KEY")
model = []  #reduce inference cost
abot = Agent(model, agent_prompts=agent_prompts)

# %%
import networkx as nx
import matplotlib.pyplot as plt

def plot_compiled_graph(compiled_graph):
    g = nx.DiGraph()

    internal_graph = compiled_graph.get_graph()
    
    # `nodes` is a set of strings
    for node in internal_graph.nodes:
        g.add_node(node)

    # `edges` is a list of tuples (source, target)
    for edge in internal_graph.edges:
        g.add_edge(edge[0], edge[1])

    pos = nx.spring_layout(g)  # layout algorithm
    plt.figure(figsize=(10, 6))
    nx.draw(g, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, arrows=True)
    plt.title("LangGraph Structure (from Compiled Graph)")
    plt.show()
# %%
# plot_compiled_graph(abot.graph)
# # %%
# messages = [HumanMessage(content="Qué carreras estudió Alex?")]
# result = abot.graph.invoke({"messages": messages})
# print(result)

# # %%
# result["messages"][4].content
# # %%
# len(result["messages"])
# # %%
