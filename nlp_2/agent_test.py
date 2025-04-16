# %%
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os
# %%
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
# %%
class Agent:

    def __init__(self, model, system=""):
        # Inicializa el agente con un modelo, un conjunto de herramientas y un 
        # mensaje del sistema opcional, y construye un grafo de estados que alterna 
        # entre invocar al LLM y ejecutar acciones.
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("regex", self.name_regex)
        graph.add_node("alex_agent", self.call_openai_alex)
        graph.add_node("max_agent", self.call_openai_max)
        graph.add_node("juan_pablo_agent", self.call_openai_juan_pablo)
        graph.add_node("concatenate_responses", self.concatenate_responses)
        graph.add_node("llm", self.call_openai_llm)
  
        graph.add_conditional_edges(
            "regex",
            self.name_regex(name="Alex"),
            {True: "alex_agent", False: END}
        )
        graph.add_conditional_edges(
            "regex",
            self.name_regex(name="Max"),
            {True: "max_agent", False: END}
        )
        graph.add_conditional_edges(
            "regex",
            self.name_regex(name="Juan Pablo"),
            {True: "juan_pablo_agent", False: END}
        )
        graph.add_edge("alex_agent", "concatenate_responses")
        graph.add_edge("max_agent", "concatenate_responses")
        graph.add_edge("juan_pablo_agent", "concatenate_responses")
        graph.add_edge("concatenate_responses", "llm")
        graph.set_entry_point("regex")
        self.graph = graph.compile()


    def name_regex(self, state: AgentState, name: str):
        # Definir regex usando state['messages'][-1].content
        content = state['messages'][-1].content
        pattern = rf"\b{name}\b"
        match = re.search(pattern, content)
        return {'name_found': bool(match)}
    
    def concatenate_responses(self, state: AgentState):
        # todo: check if this method works
        # Concatenar las respuestas de los agentes en un solo mensaje
        responses = [msg.content for msg in state['messages'] if isinstance(msg, ToolMessage)]
        concatenated_response = "\n".join(responses)
        return {'messages': [ToolMessage(content=concatenated_response)]}



    # def exists_action(self, state: AgentState):
    #     # Comprueba si el último mensaje generado por el modelo incluye
    #     #  llamadas a herramientas para decidir si debe pasar al estado de acción.
    #     result = state['messages'][-1]
    #     return len(result.tool_calls) > 0

    def call_openai_alex(self, state: AgentState):
        # todo: complete this method
        pass

    def call_openai_max(self, state: AgentState):
        # todo: complete this method
        pass

    def call_openai_juan_pablo(self, state: AgentState):
        # todo: complete this method
        pass
    
    def call_openai_llm(self, state: AgentState):
        # Envía los mensajes acumulados al modelo de lenguaje (añadiendo el mensaje del
        #  sistema si existe) y devuelve la respuesta del LLM.
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    # def take_action(self, state: AgentState):
    #     # Ejecuta las llamadas a herramientas solicitadas por el modelo, captura sus resultados y los envía de
    #     # vuelta al LLM como mensajes de herramienta.
    #     tool_calls = state['messages'][-1].tool_calls
    #     results = []
    #     for t in tool_calls:
    #         print(f"Calling: {t}")
    #         if not t['name'] in self.tools:      # check for bad tool name from LLM
    #             print("\n ....bad tool name....")
    #             result = "bad tool name, retry"  # instruct LLM to retry if bad
    #         else:
    #             result = self.tools[t['name']].invoke(t['args'])
    #         results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    #     print("Back to the model!")
    #     return {'messages': results}
# %%
# ChatOpenAI.api_key = os.getenv("OPENAI_API_KEY")
prompt = """
Eres un asistente de investigación inteligente. Utiliza el motor de búsqueda para buscar información.
Puedes realizar múltiples consultas (ya sea juntas o en secuencia).
Solo busca información cuando estés seguro de lo que necesitas.
Si necesitas buscar información antes de hacer una pregunta de seguimiento, ¡puedes hacerlo!
"""

model = []  #reduce inference cost
abot = Agent(model, system=prompt)

# %%
