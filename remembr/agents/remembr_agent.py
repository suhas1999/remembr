from typing import Annotated, Literal, Sequence, TypedDict
import traceback
import sys, re

# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


import sys, os
sys.path.append(sys.path[0] + '/..')


from remembr.utils.util import file_to_string
from remembr.tools.tools import *
from remembr.tools.functions_wrapper import FunctionsWrapper

from remembr.memory.memory import Memory

from remembr.agents.agent import Agent, AgentOutput



### Print out state of the system
def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    for k,v in state.items():
        if type(v) == str:
            print(v)

        elif type(v) == list:
            for item in v:
                if type(item) == str:
                    print(item)
                else:
                    print(item)
        else:
            print(item)

    # print(state)
    return state


def parse_json(string):
    import json as _json
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    try:
        return _json.loads(parsed)       # handles null/true/false (GPT-4o)
    except _json.JSONDecodeError:
        return eval(parsed)              # fallback for single-quoted Python dicts (Ollama)

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

def try_except_continue(state, func, max_retries=3):
    for attempt in range(max_retries):
        try:
            ret = func(state)
            return ret
        except Exception as e:
            print(f"I crashed trying to run {func} (attempt {attempt+1}/{max_retries})")
            print(e)
            traceback.print_exception(*sys.exc_info())
    raise RuntimeError(f"{func} failed after {max_retries} attempts")

class ReMEmbRAgent(Agent):

    def __init__(self, llm_type='gpt-4o', num_ctx=8192, temperature=0):

        # Wrapper that handles everything — all models go through FunctionsWrapper
        llm = self.llm_selector(llm_type, temperature, num_ctx)
        chat = FunctionsWrapper(llm)

        self.num_ctx = num_ctx
        self.temperature = temperature

        self.chat = chat
        self.llm_type = llm_type
        ### Load vectorstore
        self.embeddings = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

        # self.update_for_instance() # ref_time is None this time
        top_level_path = str(os.path.dirname(__file__)) + '/../'
        self.agent_prompt = file_to_string(top_level_path+'prompts/agent_system_prompt.txt')
        self.generate_prompt = file_to_string(top_level_path+'prompts/generate_system_prompt.txt')
        self.agent_gen_only_prompt = file_to_string(top_level_path+'prompts/agent_gen_system_prompt.txt')

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self.tool_call_log = []   # reset per query(); each entry: {tool, args, result_preview}

        self.chat_history = ChatMessageHistory()


    def llm_selector(self, llm_type, temperature, num_ctx):
        llm = None
        # OpenAI GPT models (requires OPENAI_API_KEY env var)
        if 'gpt-4' in llm_type or 'gpt-3.5' in llm_type:
            llm = ChatOpenAI(model=llm_type, temperature=temperature)

        # Support for NIMs
        elif 'nim/' in llm_type:
            llm_name = llm_type[4:]
            llm = ChatNVIDIA(model=llm_name)

        # Support for Ollama functions
        elif llm_type == 'command-r':
            llm = ChatOllama(model=llm_type, temperature=temperature, num_ctx=num_ctx)
        else:
            llm = ChatOllama(model=llm_type, format="json", temperature=temperature, num_ctx=num_ctx)

        if llm is None:
            raise Exception("No correct LLM provided")

        return llm


    def set_memory(self, memory: Memory):
        self.memory = memory
        self.create_tools(memory)
        self.build_graph()



    def create_tools(self, memory):

        template = "At time={{time}} seconds, the robot was at an average position of {{position}} with an average orientation of {{theta}} radians. "
        template += "The robot saw the following: {{page_content}}"


        class TextRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by the vector similarity-based retriever.\
                                Text embeddings of this description are used. There should always be text in here as a response! \
                                Based on the question and your context, decide what text to search for in the database. \
                                This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'.\
                                The query will then search your memories for you.")

        def _text_search(x):
            result = memory.search_by_text(x)
            self.tool_call_log.append({"tool": "retrieve_from_text", "args": {"query": x}, "result_preview": str(result)[:300]})
            return result

        self.retriever_tool = StructuredTool.from_function(
            func=_text_search,
            name="retrieve_from_text",
            description="Search and return information from your video memory in the form of captions",
            args_schema=TextRetrieverInput
        )

        class PositionRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by finding the nearest memories at this (x,y,z) position.\
                                The query must be an (x,y,z) array with floating point values, formatted as a string like '0.5, 0.2, 0.1' or '[0.5, 0.2, 0.1]'. \
                                Based on the question and your context, decide what position to search for in the database. \
                                The query will then search your memories for you.")
        def _position_search(x):
            # LLMs often pass position as a string like "[1.4, 2.3, -0.1]" — coerce to tuple
            if isinstance(x, str):
                try:
                    import ast
                    x = tuple(ast.literal_eval(x))
                except Exception:
                    pass
            result = memory.search_by_position(x)
            self.tool_call_log.append({"tool": "retrieve_from_position", "args": {"position": x}, "result_preview": str(result)[:300]})
            return result

        # position-based tool
        self.position_retriever_tool = StructuredTool.from_function(
            func=_position_search,
            name="retrieve_from_position",
            description="Search and return information from your video memory by using a position array such as (x,y,z)",
            args_schema=PositionRetrieverInput
        )

        class TimeRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by finding the nearest memories at a specific time in H:M:S format.\
                                The query must be a string containing only time. \
                                Based on the question and your context, decide what time to search for in the database. \
                                This query argument should be an HMS time such as 08:02:03 with leading zeros. \
                                The query will then search your memories for you.")

        def _time_search(x):
            result = memory.search_by_time(x)
            self.tool_call_log.append({"tool": "retrieve_from_time", "args": {"time": x}, "result_preview": str(result)[:300]})
            return result

        # time-based tool
        self.time_retriever_tool = StructuredTool.from_function(
            func=_time_search,
            name="retrieve_from_time",
            description="Search and return information from your video memory by using an H:M:S time.",
            args_schema=TimeRetrieverInput
        )

        self.tool_list = [self.retriever_tool, self.position_retriever_tool, self.time_retriever_tool]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]

    ### Nodes

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        messages = state["messages"]

        model = self.chat


        # limit to 3 tool calls.
        if self.agent_call_count < 3:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.agent_prompt
        else:
            prompt = self.agent_gen_only_prompt

        # Original prompt layout — same for all models (GPT-4o goes through FunctionsWrapper)
        agent_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                (("human"), self.previous_tool_requests),
                ("ai", prompt),
                ("human", "{question}"),
            ]
        )


        # Ollama/FunctionsWrapper JSON path can't handle ToolMessage — convert to AIMessage.
        # GPT-4o native path requires ToolMessage to follow AIMessage(tool_calls) — keep as-is.
        if not self.chat.use_gpt:
            for i in range(len(messages)):
                if type(messages[i]) == ToolMessage:
                    messages[i] = AIMessage(id=messages[i].id, content=messages[i].content)

        question = f"The question is: {messages[0].content}"   # .content = plain text
        history  = list(messages[:])

        # ── Debug: log exact prompt sent to LLM ──────────────────────────────
        formatted = agent_prompt.format_messages(question=question, chat_history=history)
        lines = [f"\n{'='*70}",
                 f"[DEBUG] agent_call={self.agent_call_count}  model={self.llm_type}"]
        for msg in formatted:
            role = msg.__class__.__name__.replace("Message", "")
            lines.append(f"  [{role}]\n{msg.content}\n")
        lines.append(f"  [Tools] {[t['name'] for t in self.tool_definitions]}")
        lines.append("="*70)
        block = "\n".join(lines)
        print(block)
        if hasattr(self, '_debug_log_path') and self._debug_log_path:
            with open(self._debug_log_path, 'a') as _f:
                _f.write(block + "\n")
        # ─────────────────────────────────────────────────────────────────────

        model = agent_prompt | model
        response = model.invoke({"question": question, "chat_history": history})

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    args = re.sub("\{.*?\}", "", str(tool_call['args'])) # remove curly braces
                    self.previous_tool_requests += f"I previously used the {tool_call['name']} tool with the arguments: {args}.\n"
                    # Log LLM decision before tool executes (result logged inside tool wrapper)
                    self.tool_call_log.append({"step": f"agent_call_{self.agent_call_count}", "tool_chosen": tool_call['name'], "args_chosen": tool_call['args']})

        self.agent_call_count += 1


        return {"messages": [response]}


    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        messages = state["messages"]
        question = messages[0].content \
                + "\n Please responsed in the desired format."
        last_message = messages[-1]


        docs = last_message.content

        prompt = PromptTemplate(
            template=self.generate_prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'question':question})


        gen_prompt = ChatPromptTemplate.from_messages(
            [
                # ("human", "What do you do?"),
                ("system", filled_prompt.text),
                MessagesPlaceholder("chat_history"),
                # ("ai", filled_prompt.text),
                ("human", "{question}"),

            ]
        )

        model = gen_prompt | self.chat

        response = model.invoke({"question": question, "chat_history": messages[1:]})

        # let us parse and check the output is a dictionary. raise error otherwise
        response = ''.join(response.content.splitlines())

        try:
            if '```json' not in response:
                # try parsing on its own since we cannot always trust llms
                try:
                    import json as _json
                    parsed = _json.loads(response)   # handles null/true/false (GPT-4o)
                except Exception:
                    parsed = eval(response)          # fallback for Ollama single-quoted dicts
            else:
                parsed = parse_json(response)

            # Unwrap nested tool_response format that the prompt template produces:
            # {"tool": "__conversational_response", "tool_input": {"response": {...}}}
            if isinstance(parsed, dict):
                if 'tool_input' in parsed:
                    inner = parsed['tool_input']
                    if isinstance(inner, dict) and 'response' in inner:
                        parsed = inner['response']
                    elif isinstance(inner, dict):
                        parsed = inner
                elif 'response' in parsed and isinstance(parsed['response'], dict):
                    parsed = parsed['response']

            # fill in any missing optional keys with None
            for key in ["time", "text", "binary", "position", "duration"]:
                parsed.setdefault(key, None)

            if type(parsed['position']) == str:
                try:
                    parsed['position'] = eval(parsed['position'])
                except Exception:
                    parsed['position'] = None

            if (parsed['position'] is not None) and len(parsed['position']) != 3:
                raise ValueError(f"Shape of position was incorrect. {parsed['position']}. Retrying...")

        except Exception as e:
            raise ValueError(f"Generate call failed: {e}")

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        return {"messages": [str(parsed)]}



    def build_graph(self):

        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))  # agent
        # retrieve = ToolNode([self.retriever_tool])
        tool_node = ToolNode(self.tool_list)
        workflow.add_node("action", tool_node)
        # workflow.add_node("action", lambda state: try_except_continue(state, tool_node))


        # workflow.add_node("action", self.call_tool)

        workflow.add_node(
            "generate", lambda state: try_except_continue(state, self.generate)
        )  # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not


        workflow.set_entry_point("agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            should_continue,
            {
                # Translate the condition outputs to nodes in our graph
                "continue": "action",
                "end": "generate",
            },
        )


        workflow.add_edge('action', 'agent')

        workflow.add_edge("generate", END)

        # Compile
        self.graph = workflow.compile()


    def query(self, question: str, debug_log_path: str = None):
        self.tool_call_log = []
        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self._debug_log_path = debug_log_path
        if debug_log_path:
            with open(debug_log_path, 'w') as _f:
                _f.write(f"Question: {question}\n\n")

        inputs = { "messages": [
                                (("user", question)),
            ]
        }

        out = self.graph.invoke(inputs)
        response = out['messages'][-1]
        response = ''.join(response.content.splitlines())

        if '```json' not in response:
            # try parsing on its own since we cannot always trust llms
            try:
                import json as _json
                parsed = _json.loads(response)   # handles null/true/false (GPT-4o)
            except Exception:
                parsed = eval(response)          # fallback for Ollama single-quoted dicts
        else:
            parsed = parse_json(response)

        response = AgentOutput.from_dict(parsed)


        return response

if __name__ == "__main__":

    from memory.milvus_memory import MilvusMemory

    # llm_name = 
    # Options: 'nim/meta/llama-3.1-405b-instruct', 'gpt-4o', or any Ollama LLMs (such as 'codestral')
    memory = MilvusMemory("test", db_ip='127.0.0.1')

    llm_name = 'gpt-4o' 
    agent = ReMEmbRAgent(llm_type=llm_name)

    agent.set_memory(memory)

    response = agent.query("Where can I sit?")
    response = agent.query_position("Where can I sit?")

