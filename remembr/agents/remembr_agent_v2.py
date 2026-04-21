"""
ReMEmbR v2 Agent.

Same three retrieval tool names as v1 (retrieve_from_text, retrieve_from_position,
retrieve_from_time) so the v1 prompts work without changes. The fourth tool,
examine_keyframes, is new — it sends the actual stored JPEG images to GPT-4o vision
and answers a visual question.

Key v2 improvements over v1:
- retrieve_from_text: hybrid BGE + SigLIP search (RRF fused) instead of pure text search
- retrieve_from_time: chronological time-range query instead of vector similarity
- All three tools return image paths so the agent can call examine_keyframes
- examine_keyframes auto-includes the stored frame before and after each requested image,
  giving the VLM a (before → main → after) triplet for motion / trajectory context
"""

import base64
import datetime
import os
import re
import sys
import traceback
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
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


# Appended to the v1 agent prompt to inform the LLM about the new 4th tool
_EXAMINE_KEYFRAMES_ADDENDUM = """
5. examine_keyframes: After ANY retrieval call that returns image paths, you should call this tool to LOOK at the actual images.
   Captions are imperfect summaries — the images contain the ground truth. Always examine images when:
   - Answering questions about position, time, or duration (verify the scene matches the caption)
   - Any visual detail matters: colors, signs, objects, people, directions
   - You need to confirm or disambiguate what a caption describes
   - The question could be answered more precisely from visual evidence than from text alone

   Pass the "Image: /path/..." paths exactly as shown in your retrieved context.
   The tool automatically includes the frame before and after each image for motion/trajectory context.

   IMPORTANT: Do not skip this tool just because a caption seems sufficient.
   Captions are compressed and lossy — examine the actual images to get precise answers.
   You MUST call a retrieval tool first to get image paths before calling examine_keyframes.
"""


# ── LangGraph state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _should_continue(state: AgentState) -> Literal["continue", "end"]:
    return "continue" if state["messages"][-1].tool_calls else "end"


def _try_except_continue(state, func, max_retries=5):
    import time as _time
    for attempt in range(max_retries):
        try:
            return func(state)
        except Exception as e:
            print(f"[WARN] {func.__name__} failed (attempt {attempt+1}/{max_retries}): {e}")
            traceback.print_exc()
            wait = 15 * (2 ** attempt)
            print(f"[WARN] retrying in {wait}s...")
            _time.sleep(wait)
    raise RuntimeError(f"{func.__name__} failed after {max_retries} attempts")


def _parse_json(string: str) -> dict:
    import json as _json
    s = "".join(string.splitlines())
    if "```json" in s:
        s = re.search(r"```json(.*?)```", s, re.DOTALL | re.IGNORECASE).group(1).strip()
    try:
        return _json.loads(s)
    except Exception:
        return eval(s)


# ── Agent class ───────────────────────────────────────────────────────────────

class ReMEmbRAgentV2(Agent):

    def __init__(self, llm_type: str = "gpt-4o", num_ctx: int = 8192 * 8, temperature: float = 0):
        self.llm_type = llm_type
        self.num_ctx = num_ctx
        self.temperature = temperature

        llm = self._make_llm(llm_type, temperature, num_ctx)
        self.chat = FunctionsWrapper(llm)

        self._siglip = None  # lazy-loaded on first hybrid search

        top_level = os.path.join(os.path.dirname(__file__), "..")

        # Use the battle-tested v1 prompts; append examine_keyframes description
        self.agent_prompt = (
            file_to_string(os.path.join(top_level, "prompts/agent_system_prompt.txt"))
            + _EXAMINE_KEYFRAMES_ADDENDUM
        )
        self.generate_prompt = file_to_string(
            os.path.join(top_level, "prompts/generate_system_prompt.txt")
        )
        self.agent_gen_only_prompt = file_to_string(
            os.path.join(top_level, "prompts/agent_gen_system_prompt.txt")
        )

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self.tool_call_log = []
        self._debug_log_path = None

    def _make_llm(self, llm_type, temperature, num_ctx):
        if "gpt" in llm_type:
            return ChatOpenAI(model=llm_type, temperature=temperature)
        if llm_type == "command-r":
            return ChatOllama(model=llm_type, temperature=temperature, num_ctx=num_ctx)
        return ChatOllama(model=llm_type, format="json", temperature=temperature, num_ctx=num_ctx)

    def _get_siglip(self):
        """Lazy-load SigLIP encoder — only instantiated if hybrid search is actually called."""
        if self._siglip is None:
            from models.siglip_encoder import SigLIPEncoder
            self._siglip = SigLIPEncoder()
        return self._siglip

    def set_memory(self, memory: MilvusMemoryV2):
        self.memory = memory
        self.create_tools(memory)
        self.build_graph()

    # ── Tool definitions ──────────────────────────────────────────────────────

    def create_tools(self, memory: MilvusMemoryV2):

        # 1. retrieve_from_text ────────────────────────────────────────────────
        # v2 improvement: internally uses hybrid BGE+SigLIP RRF instead of single embedding
        class TextRetrieverInput(BaseModel):
            x: str = Field(
                description="The query that will be searched by the vector similarity-based retriever. "
                            "Text embeddings of this description are used. There should always be text in here. "
                            "Based on the question and your context, decide what text to search for in the database. "
                            "This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'. "
                            "The query will then search your memories for you."
            )

        def _text_search(x: str) -> str:
            entries = memory.search_hybrid(x, k=3, siglip=self._get_siglip())
            result = memory.memory_to_string(entries)
            self.tool_call_log.append({
                "tool": "retrieve_from_text", "args": {"query": x},
                "result_preview": result[:300],
            })
            return result or "No relevant memories found."

        self.retriever_tool = StructuredTool.from_function(
            func=_text_search,
            name="retrieve_from_text",
            description="Search and return information from your video memory in the form of captions. "
                        "Results include Image paths that you can pass to examine_keyframes to see the actual stored images.",
            args_schema=TextRetrieverInput,
        )

        # 2. retrieve_from_position ────────────────────────────────────────────
        class PositionRetrieverInput(BaseModel):
            x: str = Field(
                description="The query that will be searched by finding the nearest memories at this (x,y,z) position. "
                            "The query must be an (x,y,z) array with floating point values, formatted as a string like '0.5, 0.2, 0.1' or '[0.5, 0.2, 0.1]'. "
                            "Based on the question and your context, decide what position to search for in the database."
            )

        def _position_search(x: str) -> str:
            if isinstance(x, str):
                try:
                    import ast
                    x = tuple(ast.literal_eval(x))
                except Exception:
                    pass
            entries = memory.search_by_position(x, k=3)
            result = memory.memory_to_string(entries)
            self.tool_call_log.append({
                "tool": "retrieve_from_position", "args": {"position": x},
                "result_preview": result[:300],
            })
            return result or "No memories found near that position."

        self.position_retriever_tool = StructuredTool.from_function(
            func=_position_search,
            name="retrieve_from_position",
            description="Search and return information from your video memory by using a position array such as (x,y,z)",
            args_schema=PositionRetrieverInput,
        )

        # 3. retrieve_from_time ────────────────────────────────────────────────
        # v2 improvement: returns all entries in the time window in chronological order
        # (instead of v1's approximate vector similarity on a time encoding)
        class TimeRetrieverInput(BaseModel):
            x: str = Field(
                description="The query that will be searched by finding memories near a specific time in H:M:S format. "
                            "The query must be a string containing only time. "
                            "Based on the question and your context, decide what time to search for in the database. "
                            "This query argument should be an HMS time such as 08:02:03 with leading zeros."
            )

        def _time_search(x: str) -> str:
            timestamp = _parse_hms_to_unix(x, memory.time_start)
            # Return entries within a tight ±10s window for local temporal context
            entries = memory.get_nearby_in_time(timestamp, window_seconds=10)
            result = memory.memory_to_string(entries)
            self.tool_call_log.append({
                "tool": "retrieve_from_time", "args": {"time": x},
                "result_preview": result[:300],
            })
            return result or "No memories found near that time."

        self.time_retriever_tool = StructuredTool.from_function(
            func=_time_search,
            name="retrieve_from_time",
            description="Get tight local context (±10 seconds) around a specific known time. "
                        "Use this ONLY after you have already identified a key event/location via retrieve_from_text or retrieve_from_position. "
                        "Do NOT use this as your primary search — always find the key point semantically first, then optionally call this to see what happened immediately before/after that moment. "
                        "Returns entries in chronological order.",
            args_schema=TimeRetrieverInput,
        )

        # 4. examine_keyframes ─────────────────────────────────────────────────
        class ExamineKeyframesInput(BaseModel):
            image_paths: str = Field(
                description="Comma-separated list of image file paths. "
                            "Copy the 'Image: /path/...' values exactly as shown in your retrieved context. "
                            "Example: '/data/v2/keyframes/0/1721829332.456.jpg,/data/v2/keyframes/0/1721829347.123.jpg'"
            )
            question: str = Field(
                description="A specific visual question about the images. "
                            "Examples: 'What color is the floor?', 'Read the text on any visible signs', "
                            "'Which direction is the robot turning?', 'Is there a water fountain visible?'"
            )

        def _examine_keyframes(image_paths: str, question: str) -> str:
            paths = [p.strip() for p in image_paths.split(",") if p.strip()]

            # For each requested image, expand to a (before → main → after) triplet.
            # Adjacent stored frames provide motion and location context for the VLM.
            labeled_images = []  # [(label, path, entry_or_None), ...]
            seen_paths = set()

            for path in paths:
                ts = _timestamp_from_path(path)

                if ts is not None:
                    entry = memory.get_entry_near_timestamp(ts)
                    if entry is not None:
                        before, after = memory.get_adjacent_entries(entry["id"])

                        if before and before.get("image_path") and before["image_path"] not in seen_paths:
                            labeled_images.append(("BEFORE (previous anchor)", before["image_path"], before))
                            seen_paths.add(before["image_path"])

                        if path not in seen_paths:
                            labeled_images.append(("MAIN FRAME", path, entry))
                            seen_paths.add(path)

                        if after and after.get("image_path") and after["image_path"] not in seen_paths:
                            labeled_images.append(("AFTER (next anchor)", after["image_path"], after))
                            seen_paths.add(after["image_path"])
                        continue

                # Fallback: no DB match, use path as-is
                if path not in seen_paths:
                    labeled_images.append(("FRAME", path, None))
                    seen_paths.add(path)

            result = _call_vision_vlm(labeled_images, question)
            self.tool_call_log.append({
                "tool": "examine_keyframes",
                "args": {"image_paths": image_paths, "question": question},
                "result_preview": result[:300],
            })
            return result

        self.examine_tool = StructuredTool.from_function(
            func=_examine_keyframes,
            name="examine_keyframes",
            description="Load stored camera images and answer a visual question about them. "
                        "Use for colors, sign text, counting objects, direction of movement, or any visual detail "
                        "that captions might have missed. Automatically includes the frame before and after each "
                        "requested image to give you motion context.",
            args_schema=ExamineKeyframesInput,
        )

        self.tool_list = [
            self.retriever_tool,
            self.position_retriever_tool,
            self.time_retriever_tool,
            self.examine_tool,
        ]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]

    # ── LangGraph nodes ───────────────────────────────────────────────────────

    def agent(self, state: AgentState) -> dict:
        messages = state["messages"]

        if not self.chat.use_gpt:
            messages = [
                AIMessage(id=m.id, content=m.content) if isinstance(m, ToolMessage) else m
                for m in messages
            ]

        if self.agent_call_count < 6:
            model = self.chat.bind_tools(tools=self.tool_definitions)
            prompt_text = self.agent_prompt
        else:
            model = self.chat
            prompt_text = self.agent_gen_only_prompt

        agent_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("human", self.previous_tool_requests),
            ("ai", prompt_text),
            ("human", "{question}"),
        ])

        question = f"The question is: {messages[0].content}"
        formatted = agent_prompt.format_messages(question=question, chat_history=list(messages))

        # Debug log
        lines = [f"\n{'='*70}", f"[DEBUG] agent_call={self.agent_call_count}  model={self.llm_type}"]
        for msg in formatted:
            role = msg.__class__.__name__.replace("Message", "")
            lines.append(f"  [{role}]\n{msg.content}\n")
        lines.append(f"  [Tools] {[t['name'] for t in self.tool_definitions]}")
        lines.append("=" * 70)
        block = "\n".join(lines)
        print(block)
        if self._debug_log_path:
            with open(self._debug_log_path, "a") as f:
                f.write(block + "\n")

        chain = agent_prompt | model
        response = chain.invoke({"question": question, "chat_history": list(messages)})

        if response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] != "__conversational_response":
                    args = re.sub(r"\{.*?\}", "", str(tc["args"]))
                    self.previous_tool_requests += (
                        f"I previously used the {tc['name']} tool with the arguments: {args}.\n"
                    )
                    self.tool_call_log.append({
                        "step": f"agent_call_{self.agent_call_count}",
                        "tool_chosen": tc["name"],
                        "args_chosen": tc["args"],
                    })

        self.agent_call_count += 1
        return {"messages": [response]}

    def generate(self, state: AgentState) -> dict:
        messages = state["messages"]
        question = messages[0].content + "\n Please respond in the desired format."

        prompt = PromptTemplate(
            template=self.generate_prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({"question": question})

        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", filled_prompt.text),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        chain = gen_prompt | self.chat
        response = chain.invoke({"question": question, "chat_history": list(messages[1:])})

        raw = "".join(response.content.splitlines())
        try:
            parsed = _parse_json(raw)

            # Unwrap nested tool_response format
            if isinstance(parsed, dict):
                if "tool_input" in parsed:
                    inner = parsed["tool_input"]
                    parsed = inner.get("response", inner) if isinstance(inner, dict) else inner
                elif "response" in parsed and isinstance(parsed["response"], dict):
                    parsed = parsed["response"]

            for key in ["time", "text", "binary", "position", "duration"]:
                if key not in parsed:
                    raise ValueError(f"Missing key '{key}' in generate response")

            if isinstance(parsed.get("position"), str):
                try:
                    parsed["position"] = eval(parsed["position"])
                except Exception:
                    parsed["position"] = None
            if parsed.get("position") is not None and len(parsed["position"]) != 3:
                raise ValueError(f"Bad position shape: {parsed['position']}")

        except Exception as e:
            raise ValueError(f"Generate failed: {e}")

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        return {"messages": [str(parsed)]}

    # ── Graph ─────────────────────────────────────────────────────────────────

    def build_graph(self):
        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", lambda s: _try_except_continue(s, self.agent))
        workflow.add_node("action", ToolNode(self.tool_list))
        workflow.add_node("generate", lambda s: _try_except_continue(s, self.generate))

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
        self.tool_call_log = []
        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self._debug_log_path = debug_log_path

        if debug_log_path:
            with open(debug_log_path, "w") as f:
                f.write(f"Question: {question}\n\n")

        out = self.graph.invoke({"messages": [("user", question)]})
        raw = "".join(out["messages"][-1].content.splitlines())
        parsed = _parse_json(raw)
        return AgentOutput.from_dict(parsed)


# ── Utility functions ─────────────────────────────────────────────────────────

def _parse_hms_to_unix(time_str: str, reference_unix: float = None) -> float:
    """Parse HH:MM:SS (or full datetime) to a unix timestamp."""
    import time as _time
    time_str = time_str.strip()
    from time import strftime, localtime as _localtime

    ref_local = _localtime(reference_unix) if reference_unix else _localtime()
    mdy = strftime("%m/%d/%Y", ref_local)

    for fmt in ("%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(time_str, fmt)
            if fmt == "%H:%M:%S":
                # Inject the reference date so the timestamp lands in the right day
                dt = dt.replace(year=ref_local.tm_year, month=ref_local.tm_mon, day=ref_local.tm_mday)
            return dt.timestamp()
        except ValueError:
            continue

    # Last resort
    try:
        return datetime.datetime.strptime(mdy + " " + time_str, "%m/%d/%Y %H:%M:%S").timestamp()
    except ValueError:
        raise ValueError(f"Cannot parse time: {time_str!r}")


def _timestamp_from_path(path: str) -> float:
    """Extract unix timestamp from a keyframe filename like 1721829332.456000.jpg"""
    basename = os.path.basename(path)
    stem = os.path.splitext(basename)[0]
    # Remove any suffix like "_revisit"
    stem = stem.split("_")[0]
    try:
        return float(stem)
    except ValueError:
        return None


def _call_vision_vlm(labeled_images: list, question: str, model: str = "gpt-4o") -> str:
    """
    Send labeled images to GPT-4o and answer a visual question.

    labeled_images: [(label, path, entry_or_None), ...]
    entry contains time, position, caption — sent as text context alongside each image.
    """
    from openai import OpenAI
    from time import strftime, localtime
    client = OpenAI()

    intro = (
        "You are examining frames from a robot's camera. "
        "You will see frames labeled BEFORE (previous anchor), MAIN FRAME (the key frame), "
        "and AFTER (next anchor), giving you temporal/motion context around the main frame. "
        "Each frame is accompanied by its timestamp, position, and caption.\n\n"
        f"Please answer this question about the images: {question}\n\n"
        "Be specific. Use the timestamps and positions to reason about timing and location."
    )
    content = [{"type": "text", "text": intro}]

    loaded = 0
    for label, path, entry in labeled_images:
        if not os.path.exists(path):
            continue
        meta_parts = [f"[{label}]"]
        if entry:
            t_str = strftime("%Y-%m-%d %H:%M:%S", localtime(entry["time"]))
            pos = [round(v, 2) for v in entry.get("position", [0, 0, 0])]
            caption = entry.get("caption", "")[:300]
            meta_parts.append(f"Time: {t_str} | Position: {pos}")
            meta_parts.append(f"Caption: {caption}")
        content.append({"type": "text", "text": "\n".join(meta_parts)})
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        loaded += 1

    if loaded == 0:
        return f"No valid image files found. Provided paths: {[p for _, p, *_ in labeled_images]}"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        temperature=0,
    )
    return response.choices[0].message.content
