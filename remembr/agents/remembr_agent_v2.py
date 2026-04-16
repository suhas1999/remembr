"""
ReMEmbR v2 Agent.

Four general-purpose tools instead of three type-specific ones:
  1. search_memory         — hybrid BGE + SigLIP retrieval with context expansion
  2. search_near_position  — GPS position search
  3. get_nearby_in_time    — chronological time-window retrieval
  4. examine_keyframes     — sends actual stored images to GPT-4o and asks a visual question

No question-type routing: the LLM reads tool descriptions and decides its own strategy.
This enables the system to handle visual detail questions (colors, signs, gestures) that
ReMEmbR v1 could not answer because it never stored images.
"""

import base64
import datetime
import os
import re
import sys
import time
import traceback
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.agent import Agent, AgentOutput
from memory.milvus_memory_v2 import MilvusMemoryV2
from tools.functions_wrapper import FunctionsWrapper
from utils.util import file_to_string


# ── LangGraph state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _should_continue(state: AgentState) -> Literal["continue", "end"]:
    last = state["messages"][-1]
    return "continue" if last.tool_calls else "end"


def _try_except_continue(state, func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func(state)
        except Exception as e:
            print(f"[WARN] {func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}")
            traceback.print_exc()
    raise RuntimeError(f"{func.__name__} failed after {max_retries} attempts")


def _parse_json(string: str) -> dict:
    import json as _json
    cleaned = "".join(string.splitlines())
    if "```json" in cleaned:
        cleaned = re.search(r"```json(.*?)```", cleaned, re.DOTALL | re.IGNORECASE).group(1).strip()
    try:
        return _json.loads(cleaned)
    except Exception:
        return eval(cleaned)


# ── Agent class ───────────────────────────────────────────────────────────────

class ReMEmbRAgentV2(Agent):

    MAX_TOOL_CALLS = 5  # v2 has 4 tools and more complex questions — allow more steps

    def __init__(self, llm_type: str = "gpt-4o", temperature: float = 0, num_ctx: int = 8192 * 8):
        self.llm_type = llm_type
        self.temperature = temperature
        self.num_ctx = num_ctx

        llm = self._make_llm(llm_type, temperature, num_ctx)
        self.chat = FunctionsWrapper(llm)

        # Lazy SigLIP — only loaded when search_memory is actually called with visual queries
        self._siglip = None

        top_level = os.path.join(os.path.dirname(__file__), "..")
        self.agent_prompt = file_to_string(os.path.join(top_level, "prompts/v2/agent_system.txt"))
        self.generate_prompt = file_to_string(os.path.join(top_level, "prompts/v2/generate_answer.txt"))

        self._reset_call_state()

    def _make_llm(self, llm_type, temperature, num_ctx):
        if "gpt" in llm_type:
            return ChatOpenAI(model=llm_type, temperature=temperature)
        return ChatOllama(model=llm_type, format="json", temperature=temperature, num_ctx=num_ctx)

    def _reset_call_state(self):
        self.agent_call_count = 0
        self.tool_call_log = []
        self._previous_tools_summary = "Tools used so far: (none)\n"

    def set_memory(self, memory: MilvusMemoryV2):
        self.memory = memory
        self._build_tools(memory)
        self._build_graph()

    # ── Tool definitions ──────────────────────────────────────────────────────

    def _build_tools(self, memory: MilvusMemoryV2):

        # 1. search_memory ─────────────────────────────────────────────────────
        class SearchMemoryInput(BaseModel):
            query: str = Field(
                description="Text phrase to search for in memory. Use natural language descriptions "
                            "of what you are looking for. Examples: 'water fountain', 'red exit sign', "
                            "'glass doors leading outside', 'person sitting on bench'."
            )

        def _search_memory(query: str) -> str:
            entries = memory.search_hybrid(query, k=5, siglip=self._get_siglip())
            result = memory.format_entries_with_context(entries, max_neighbors=1)
            self.tool_call_log.append({
                "tool": "search_memory", "args": {"query": query},
                "result_preview": result[:400],
            })
            return result if result.strip() else "No relevant memories found for that query."

        # 2. search_near_position ──────────────────────────────────────────────
        class SearchPositionInput(BaseModel):
            x: float = Field(description="X coordinate in meters")
            y: float = Field(description="Y coordinate in meters")
            z: float = Field(description="Z coordinate in meters")

        def _search_near_position(x: float, y: float, z: float) -> str:
            entries = memory.search_by_position((x, y, z), k=4)
            result = memory.format_entries_with_context(entries, max_neighbors=0)
            self.tool_call_log.append({
                "tool": "search_near_position", "args": {"x": x, "y": y, "z": z},
                "result_preview": result[:400],
            })
            return result if result.strip() else "No memories found near that position."

        # 3. get_nearby_in_time ────────────────────────────────────────────────
        class GetNearbyInTimeInput(BaseModel):
            time_str: str = Field(
                description="Time to search around in HH:MM:SS format (e.g., '10:15:32'). "
                            "Use timestamps from previous search results."
            )
            window_seconds: float = Field(
                default=120.0,
                description="How many seconds before and after the given time to include. "
                            "Use 300 for a 5-minute window, 900 for 15 minutes, 1800 for 30 minutes.",
            )

        def _get_nearby_in_time(time_str: str, window_seconds: float = 120.0) -> str:
            timestamp = _parse_time(time_str, memory.time_start)
            entries = memory.get_nearby_in_time(timestamp, window_seconds=window_seconds)
            result = _format_temporal_sequence(entries)
            self.tool_call_log.append({
                "tool": "get_nearby_in_time",
                "args": {"time_str": time_str, "window_seconds": window_seconds},
                "result_preview": result[:400],
            })
            return result if result.strip() else "No memories found in that time window."

        # 4. examine_keyframes ─────────────────────────────────────────────────
        class ExamineKeyframesInput(BaseModel):
            image_paths: str = Field(
                description="Comma-separated list of image file paths from previous search results. "
                            "Copy the 'Image path:' values exactly as shown. "
                            "Example: '/home/suhas/remembr/data/v2/keyframes/0/1721829332.456.jpg,"
                            "/home/suhas/remembr/data/v2/keyframes/0/1721829347.123.jpg'"
            )
            question: str = Field(
                description="A specific visual question about the images. "
                            "Examples: 'What color is the floor?', 'Read any visible signs', "
                            "'Which direction is the robot turning across this sequence?', "
                            "'Is there a water fountain visible?'"
            )

        def _examine_keyframes(image_paths: str, question: str) -> str:
            paths = [p.strip() for p in image_paths.split(",") if p.strip()]
            result = _call_vision_llm(paths, question, model=self.llm_type)
            self.tool_call_log.append({
                "tool": "examine_keyframes",
                "args": {"image_paths": image_paths, "question": question},
                "result_preview": result[:400],
            })
            return result

        self.tool_list = [
            StructuredTool.from_function(
                func=_search_memory, name="search_memory",
                description="Search your visual memories by text description. Returns captions, "
                            "timestamps, GPS positions, and image paths. Also returns adjacent frames "
                            "for motion context. Use this first for most questions.",
                args_schema=SearchMemoryInput,
            ),
            StructuredTool.from_function(
                func=_search_near_position, name="search_near_position",
                description="Find memories near a GPS (x, y, z) position. "
                            "Use when you know an approximate location and want nearby memories.",
                args_schema=SearchPositionInput,
            ),
            StructuredTool.from_function(
                func=_get_nearby_in_time, name="get_nearby_in_time",
                description="Get all stored memories within a time window, returned in chronological order. "
                            "Essential for duration questions ('how long were you at X'), "
                            "sequence questions ('what did you see next'), and trajectory questions.",
                args_schema=GetNearbyInTimeInput,
            ),
            StructuredTool.from_function(
                func=_examine_keyframes, name="examine_keyframes",
                description="Load actual stored images and answer a visual question about them. "
                            "Use for colors, sign text, counting objects, direction of movement, "
                            "or any visual detail that captions might have missed.",
                args_schema=ExamineKeyframesInput,
            ),
        ]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]

    def _get_siglip(self):
        """Lazy-load SigLIP encoder only when actually needed."""
        if self._siglip is None:
            from models.siglip_encoder import SigLIPEncoder
            self._siglip = SigLIPEncoder()
        return self._siglip

    # ── LangGraph nodes ───────────────────────────────────────────────────────

    def _agent_node(self, state: AgentState) -> dict:
        messages = state["messages"]

        if not self.chat.use_gpt:
            messages = _coerce_tool_messages(messages)

        if self.agent_call_count < self.MAX_TOOL_CALLS:
            model = self.chat.bind_tools(tools=self.tool_definitions)
        else:
            model = self.chat  # no tools — must answer now

        prompt = ChatPromptTemplate.from_messages([
            ("ai", self.agent_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        question = f"The question is: {messages[0].content}"
        chain = prompt | model
        response = chain.invoke({"question": question, "chat_history": list(messages)})

        if response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] != "__conversational_response":
                    self.tool_call_log.append({
                        "step": f"agent_call_{self.agent_call_count}",
                        "tool_chosen": tc["name"],
                        "args_chosen": tc["args"],
                    })
                    self._previous_tools_summary += f"  - {tc['name']}({tc['args']})\n"

        self.agent_call_count += 1
        return {"messages": [response]}

    def _generate_node(self, state: AgentState) -> dict:
        messages = state["messages"]
        question = messages[0].content

        # Build the filled prompt with the question injected
        filled_prompt = self.generate_prompt.replace("{question}", question)

        prompt = ChatPromptTemplate.from_messages([
            ("system", filled_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Based on all the context above, answer: {question}"),
        ])

        chain = prompt | self.chat
        response = chain.invoke({
            "question": question,
            "chat_history": list(messages[1:]),
        })

        raw = "".join(response.content.splitlines())
        parsed = _parse_json(raw)

        # Unwrap nested tool_response format if present
        if isinstance(parsed, dict):
            if "tool_input" in parsed:
                inner = parsed["tool_input"]
                parsed = inner.get("response", inner)
            elif "response" in parsed and isinstance(parsed["response"], dict):
                parsed = parsed["response"]

        required_keys = ["type", "text", "binary", "position", "time", "duration"]
        for key in required_keys:
            if key not in parsed:
                raise ValueError(f"Generate node: missing key '{key}' in response")

        self._reset_call_state()
        return {"messages": [str(parsed)]}

    # ── Graph ─────────────────────────────────────────────────────────────────

    def _build_graph(self):
        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", lambda s: _try_except_continue(s, self._agent_node))
        workflow.add_node("action", ToolNode(self.tool_list))
        workflow.add_node("generate", lambda s: _try_except_continue(s, self._generate_node))

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", _should_continue, {
            "continue": "action",
            "end": "generate",
        })
        workflow.add_edge("action", "agent")
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()

    # ── Public interface ──────────────────────────────────────────────────────

    def query(self, question: str, debug_log_path: str = None) -> AgentOutput:
        self._reset_call_state()
        self._debug_log_path = debug_log_path

        if debug_log_path:
            with open(debug_log_path, "w") as f:
                f.write(f"Question: {question}\n\n")

        out = self.graph.invoke({"messages": [("user", question)]})
        raw = "".join(out["messages"][-1].content.splitlines())
        parsed = _parse_json(raw)
        return AgentOutput.from_dict(parsed)


# ── Utility functions ─────────────────────────────────────────────────────────

def _coerce_tool_messages(messages):
    """Ollama JSON path can't handle ToolMessage — convert to AIMessage."""
    result = list(messages)
    for i, msg in enumerate(result):
        if isinstance(msg, ToolMessage):
            result[i] = AIMessage(id=msg.id, content=msg.content)
    return result


def _parse_time(time_str: str, reference_time: float = None) -> float:
    """Parse HH:MM:SS (or full datetime) to unix timestamp."""
    time_str = time_str.strip()
    for fmt in ("%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(time_str, fmt)
            if fmt == "%H:%M:%S" and reference_time is not None:
                ref = datetime.datetime.fromtimestamp(reference_time)
                dt = dt.replace(year=ref.year, month=ref.month, day=ref.day)
            return dt.timestamp()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time string: {time_str!r}")


def _format_temporal_sequence(entries: list) -> str:
    """Format a chronological list of entries for the agent, showing duration info."""
    if not entries:
        return "No entries found in that time window."
    from time import strftime, localtime
    import numpy as np
    lines = [f"Chronological sequence ({len(entries)} entries):"]
    for e in entries:
        t_str = strftime("%Y-%m-%d %H:%M:%S", localtime(e["time"]))
        pos = np.round(e.get("position", [0, 0, 0]), 3).tolist()
        revisit = " [REVISIT]" if e.get("is_revisit", 0) > 0 else ""
        lines.append(f"  {t_str}  pos={pos}{revisit}")
        lines.append(f"    {e.get('caption', '')[:200]}")
        if e.get("image_path"):
            lines.append(f"    Image path: {e['image_path']}")
    if len(entries) >= 2:
        duration = entries[-1]["time"] - entries[0]["time"]
        lines.append(f"\nSpan: {duration:.0f} seconds ({duration/60:.1f} minutes)")
        lines.append(f"First: {strftime('%H:%M:%S', localtime(entries[0]['time']))}")
        lines.append(f"Last:  {strftime('%H:%M:%S', localtime(entries[-1]['time']))}")
    return "\n".join(lines)


def _call_vision_llm(image_paths: list, question: str, model: str = "gpt-4o") -> str:
    """Send stored keyframe images to GPT-4o and answer a visual question."""
    from openai import OpenAI
    client = OpenAI()

    content = [
        {
            "type": "text",
            "text": (
                f"You are examining frames from a robot's camera. "
                f"Please answer this question about the image(s): {question}\n\n"
                f"Be specific and concise. If you see multiple images, describe what changes across them."
            ),
        }
    ]

    loaded_count = 0
    for path in image_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
            loaded_count += 1

    if loaded_count == 0:
        return f"No valid image paths found. Provided: {image_paths}"

    # Use gpt-4o for vision, regardless of what the agent LLM is
    vision_model = model if "gpt" in model else "gpt-4o"

    response = client.chat.completions.create(
        model=vision_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        temperature=0,
    )
    return response.choices[0].message.content
