#!/usr/bin/env python3
"""
Milestone 6 - Part 2: Multi-Tool Agent Controller
===================================================
MLOps Course - Module 7

A ReAct-style agent (Reason + Act) that autonomously selects between:
  • retriever  – searches the ChromaDB vector store from Part 1
  • summarizer – condenses long retrieved text via the LLM
  • answer     – synthesises a final grounded answer and stops

Usage:
    # Run standalone (imports RAG components from rag_pipeline.py)
    python agent_controller.py

    # Or import and use in a notebook:
    from agent_controller import build_agent, AGENT_TASKS
    agent = build_agent()
    trace = agent.run(AGENT_TASKS[0]["task"])
"""

import os, sys, json, time, logging, warnings
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Import RAG components ─────────────────────────────────────────────────────
from rag_pipeline import (
    DOCUMENTS, CONFIG,
    chunk_documents, EmbeddingModel, VectorStore, LLMGenerator, RAGPipeline,
)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

class RetrieverTool:
    """
    Tool 1 – Retriever
    Wraps the Part-1 VectorStore.retrieve() as an agent-callable tool.
    Returns structured results (title + content snippet + relevance score).

    Trigger policy: agent uses this tool when the task requires finding facts,
    definitions, comparisons, or any information that may reside in the corpus.
    """
    name = "retriever"
    description = (
        "Searches the knowledge base for relevant information. "
        "Input: a search query string. "
        "Use when you need to find facts, explanations, or comparisons."
    )

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def run(self, query: str, top_k: int = 3) -> Dict:
        t0 = time.time()
        hits, latency = self.vs.retrieve(query, top_k)
        return {
            "tool":    self.name,
            "query":   query,
            "latency_s": latency,
            "results": [
                {
                    "title":           h["metadata"]["title"],
                    "doc_id":          h["metadata"]["doc_id"],
                    "relevance_score": h["relevance_score"],
                    "snippet":         h["content"][:400],
                }
                for h in hits
            ],
        }


class SummarizerTool:
    """
    Tool 2 – Summarizer
    Uses the LLM to condense retrieved text into 3-5 concise sentences.

    Trigger policy: agent uses this tool after retrieval when the retrieved
    passages are long and need distillation before composing a final answer,
    or when the task explicitly asks for a summary.
    """
    name = "summarizer"
    description = (
        "Summarises long text into 3-5 key sentences. "
        "Input: the text to summarise. "
        "Use after retrieval when passages are lengthy or when the task asks for a summary."
    )

    def __init__(self, llm: LLMGenerator):
        self.llm = llm

    def run(self, text: str) -> Dict:
        prompt = (
            "<s>[INST] Summarise the following text in 4-5 concise sentences, "
            f"preserving key facts:\n\n{text[:1500]} [/INST]"
        )
        t0 = time.time()
        out = self.llm.pipe(prompt, max_new_tokens=200)[0]["generated_text"]
        latency = round(time.time() - t0, 3)
        summary = out.split("[/INST]")[-1].strip()
        return {"tool": self.name, "summary": summary, "latency_s": latency}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class AgentController:
    """
    ReAct-style agent controller.

    Decision loop (up to max_steps):
      1. LLM is given the task + recent action history → decides next action as JSON.
      2. Action is executed (retriever / summarizer / answer).
      3. Result is appended to history.
      4. Repeat until action == "answer" or max_steps reached.

    Tool-selection policy:
      • retriever  → default first action for any information-seeking task.
      • summarizer → used after retrieval when >400 words of context exist.
      • answer     → used when history contains sufficient context to respond.

    Full trace is logged and returned for transparency.
    """

    SYSTEM_PROMPT = """You are an AI agent that solves tasks using tools.

Available tools:
  - retriever: Searches a knowledge base for relevant information. Use this first for any factual question.
  - summarizer: Summarises long retrieved text. Use after retrieval when text is lengthy.
  - answer: Provide the final answer using gathered information. Use when you have enough context.

Respond ONLY with valid JSON (no extra text, no markdown):
{"action": "<retriever|summarizer|answer>", "input": "<your query or text>", "reasoning": "<why you chose this action>"}"""

    def __init__(
        self,
        retriever_tool: RetrieverTool,
        summarizer_tool: SummarizerTool,
        llm: LLMGenerator,
        max_steps: int = 5,
    ):
        self.tools = {
            "retriever":  retriever_tool,
            "summarizer": summarizer_tool,
        }
        self.llm = llm
        self.max_steps = max_steps

    # ── decision ─────────────────────────────────────────────────────────────
    def _decide(self, task: str, history: List[Dict]) -> Dict:
        """Ask the LLM what to do next; parse JSON response."""
        hist_str = json.dumps(history[-4:], indent=2) if history else "[]"
        prompt = (
            f"<s>[INST] {self.SYSTEM_PROMPT}\n\n"
            f"Task: {task}\n\n"
            f"Action history (most recent last):\n{hist_str}\n\n"
            "What is your next action? [/INST]"
        )
        t0 = time.time()
        raw = self.llm.pipe(prompt, max_new_tokens=150)[0]["generated_text"]
        decision_latency = round(time.time() - t0, 3)
        text = raw.split("[/INST]")[-1].strip()

        # Strip markdown code fences if present
        for fence in ("```json", "```"):
            if fence in text:
                text = text.split(fence)[1].split("```")[0].strip()
                break

        try:
            decision = json.loads(text)
        except json.JSONDecodeError:
            # Robust fallback heuristic
            if not history:
                decision = {"action": "retriever", "input": task,
                            "reasoning": "Default: retrieve first"}
            elif any(s.get("action") == "retriever" for s in history):
                decision = {"action": "answer",    "input": task,
                            "reasoning": "Fallback: already retrieved, answer now"}
            else:
                decision = {"action": "retriever", "input": task,
                            "reasoning": "Fallback: try retrieval"}

        decision["decision_latency_s"] = decision_latency
        return decision

    # ── run ──────────────────────────────────────────────────────────────────
    def run(self, task: str) -> Dict:
        """Execute the agent loop; return a full observable trace."""
        trace = {
            "task":         task,
            "steps":        [],
            "final_answer": None,
            "status":       "running",
            "total_latency_s": 0.0,
        }
        history: List[Dict] = []
        t_start = time.time()

        for step_num in range(1, self.max_steps + 1):
            decision = self._decide(task, history)
            action   = decision.get("action", "answer")
            inp      = decision.get("input", task)
            reasoning = decision.get("reasoning", "")
            d_lat    = decision.get("decision_latency_s", 0.0)

            step: Dict = {
                "step":              step_num,
                "action":            action,
                "input":             inp,
                "reasoning":         reasoning,
                "decision_latency_s": d_lat,
                "tool_result":       None,
                "tool_latency_s":    None,
            }

            if action == "answer":
                # Build context from all previous retrieval results
                ctx_parts = []
                for s in trace["steps"]:
                    tr = s.get("tool_result")
                    if isinstance(tr, dict):
                        if tr.get("tool") == "retriever":
                            ctx_parts.extend(r["snippet"] for r in tr.get("results", []))
                        elif tr.get("tool") == "summarizer":
                            ctx_parts.append(tr.get("summary", ""))
                context = "\n\n".join(ctx_parts) if ctx_parts else inp

                ans, gen_lat = self.llm.generate(task, context)
                step["tool_result"]   = ans
                step["tool_latency_s"] = gen_lat
                trace["final_answer"] = ans
                trace["status"]       = "success"
                trace["steps"].append(step)
                break

            elif action in self.tools:
                result = self.tools[action].run(inp)
                step["tool_result"]   = result
                step["tool_latency_s"] = result.get("latency_s", 0.0)
                # Add compact history entry
                if action == "retriever":
                    hist_entry = {
                        "step": step_num, "action": action,
                        "retrieved_titles": [r["title"] for r in result.get("results", [])],
                    }
                else:
                    hist_entry = {
                        "step": step_num, "action": action,
                        "summary_snippet": result.get("summary", "")[:120],
                    }
                history.append(hist_entry)

            else:
                step["tool_result"] = f"Unknown action: {action}"
                trace["status"]     = "error"

            trace["steps"].append(step)

        else:
            # Exceeded max_steps – force answer
            trace["status"]       = "max_steps_reached"
            trace["final_answer"] = "Agent exceeded max steps without converging."

        trace["total_latency_s"] = round(time.time() - t_start, 3)
        return trace


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION TASKS
# ══════════════════════════════════════════════════════════════════════════════

AGENT_TASKS = [
    {"id": "task_01", "task": "What is RAG, and how does it differ from fine-tuning a language model?",
     "expected_tools": ["retriever", "answer"], "category": "comparison"},
    {"id": "task_02", "task": "Summarise the key differences between BERT and GPT architectures.",
     "expected_tools": ["retriever", "summarizer", "answer"], "category": "summarization"},
    {"id": "task_03", "task": "Explain the self-attention formula and why scaling by sqrt(dk) is important.",
     "expected_tools": ["retriever", "answer"], "category": "technical-explanation"},
    {"id": "task_04", "task": "List the best MLOps practices for deploying a machine-learning model to production.",
     "expected_tools": ["retriever", "answer"], "category": "list-generation"},
    {"id": "task_05", "task": "What challenges arise when running large language models in production, and how can they be mitigated?",
     "expected_tools": ["retriever", "summarizer", "answer"], "category": "problem-solution"},
    {"id": "task_06", "task": "Compare FAISS, Chroma, and Weaviate as vector databases. Which is best for a small research project?",
     "expected_tools": ["retriever", "answer"], "category": "comparison"},
    {"id": "task_07", "task": "Explain LoRA and QLoRA. When would you choose one over the other?",
     "expected_tools": ["retriever", "answer"], "category": "technical-explanation"},
    {"id": "task_08", "task": "What evaluation metrics should I use to measure the quality of a RAG pipeline?",
     "expected_tools": ["retriever", "answer"], "category": "evaluation"},
    {"id": "task_09", "task": "Explain the Transformer architecture to someone who has never studied deep learning.",
     "expected_tools": ["retriever", "summarizer", "answer"], "category": "beginner-explanation"},
    {"id": "task_10", "task": "How do vector embeddings and vector databases work together to enable semantic search in a RAG system?",
     "expected_tools": ["retriever", "retriever", "answer"], "category": "multi-hop"},
]


# ══════════════════════════════════════════════════════════════════════════════
# BUILD + RUN
# ══════════════════════════════════════════════════════════════════════════════

def build_agent() -> AgentController:
    """Initialise all components and return a ready-to-use AgentController."""
    print("\n[Agent] Loading embedding model …")
    embedder = EmbeddingModel(CONFIG["embedding_model"])

    print("[Agent] Building vector store …")
    vs = VectorStore(CONFIG["chroma_collection"] + "_agent", "./chroma_db_agent", embedder)
    vs.index(chunk_documents(DOCUMENTS, CONFIG["chunk_size"], CONFIG["chunk_overlap"]))

    print("[Agent] Loading LLM …")
    llm = LLMGenerator(CONFIG["llm_model"])

    retriever  = RetrieverTool(vs)
    summarizer = SummarizerTool(llm)
    agent = AgentController(retriever, summarizer, llm, max_steps=5)
    print("[Agent] Agent ready.\n")
    return agent


def evaluate_agent(agent: AgentController, tasks: List[Dict], output_dir: str = "agent_traces") -> List[Dict]:
    """Run all tasks, save individual trace JSON files, return summary."""
    os.makedirs(output_dir, exist_ok=True)
    summaries = []

    for item in tasks:
        print(f"\n{'='*65}")
        print(f"[{item['id']}] {item['task']}")
        print("=" * 65)

        trace = agent.run(item["task"])
        trace["task_id"]   = item["id"]
        trace["category"]  = item["category"]
        trace["expected_tools"] = item["expected_tools"]

        # Persist trace
        path = os.path.join(output_dir, f"{item['id']}.json")
        with open(path, "w") as f:
            json.dump(trace, f, indent=2)

        actions = [s["action"] for s in trace["steps"]]
        print(f"  Steps : {actions}")
        print(f"  Status: {trace['status']}")
        print(f"  Time  : {trace['total_latency_s']}s")
        if trace["final_answer"]:
            print(f"  Answer: {trace['final_answer'][:180]} …")

        summaries.append({
            "task_id":      item["id"],
            "category":     item["category"],
            "task":         item["task"],
            "status":       trace["status"],
            "actions":      actions,
            "steps_taken":  len(trace["steps"]),
            "total_latency_s": trace["total_latency_s"],
            "answer_snippet": (trace["final_answer"] or "")[:150],
        })

    # Overall summary
    success_rate = sum(1 for s in summaries if s["status"] == "success") / len(summaries)
    avg_latency  = sum(s["total_latency_s"] for s in summaries) / len(summaries)
    print(f"\n{'='*65}")
    print(f"AGENT EVALUATION SUMMARY")
    print(f"  Tasks evaluated : {len(summaries)}")
    print(f"  Success rate    : {success_rate*100:.0f}%")
    print(f"  Avg latency     : {avg_latency:.1f}s")
    print("=" * 65)

    with open(os.path.join(output_dir, "_summary.json"), "w") as f:
        json.dump({"success_rate": success_rate, "avg_latency_s": round(avg_latency, 2),
                   "tasks": summaries}, f, indent=2)
    return summaries


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("MILESTONE 6 – PART 2: AGENT CONTROLLER")
    print("=" * 65)
    agent = build_agent()
    evaluate_agent(agent, AGENT_TASKS, output_dir="agent_traces")
    print("\n✓ All traces saved to agent_traces/")
