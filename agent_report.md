# Agent Controller Report
## Milestone 6 – Part 2 | MLOps Course – Module 7

---

## 1. Agent Architecture Overview

The agent follows the **ReAct (Reason + Act)** pattern: at each step, the LLM is prompted to reason about the current state of the task and select the most appropriate tool, outputting its decision as structured JSON. The agent loops until it selects the `answer` action or exhausts `max_steps = 5`.

**Tools available:**

| Tool | Description | Trigger condition |
|---|---|---|
| `retriever` | Searches ChromaDB vector store (top-3 chunks) | Any task requiring factual information from the corpus |
| `summarizer` | Condenses retrieved passages via LLM | After retrieval, when passage length > ~400 words or task explicitly requests a summary |
| `answer` | Generates final grounded answer and halts | When sufficient context has been gathered |

**LLM in the loop:** `mistralai/Mistral-7B-Instruct-v0.3` (4-bit NF4, Colab T4 GPU) is used for both tool-selection decisions and final answer generation, making every evaluated run a true open-weight 7B model workflow.

---

## 2. Tool Selection Policy

The decision prompt is structured so the LLM reasons through three questions:

1. *Do I need information from the knowledge base?* → `retriever`
2. *Do I have retrieved text that is too long to use directly?* → `summarizer`
3. *Do I have enough context to answer the task?* → `answer`

The policy is implemented as a few-shot system prompt that explicitly lists each tool's trigger condition. This makes the decision logic **observable** (every decision is logged in the trace as `reasoning`) and **controllable** (the prompt can be updated to change behaviour without rewriting code).

**Practical behaviour observed:**
- For factual single-hop tasks (e.g., task_01, task_04, task_08), the agent takes 2 steps: `retriever → answer`.
- For tasks requesting a summary (task_02, task_05, task_09), the agent takes 3 steps: `retriever → summarizer → answer`.
- For multi-hop tasks (task_10), the agent issues two retrieval calls with different queries before answering.

---

## 3. Retrieval Integration

The `RetrieverTool` is a direct wrapper around the `VectorStore.retrieve()` function from Part 1, making the component fully reusable. It accepts a natural-language query string, runs HNSW approximate nearest-neighbour search, and returns a structured dict containing title, doc_id, relevance score, and a 400-character snippet for each of the top-3 results.

The agent's LLM sees these snippets in its action history and can re-query with a refined search term if the initial retrieval is insufficient (observed in task_10).

---

## 4. Performance Analysis — 10 Evaluation Tasks

| Task ID | Category | Steps | Actions taken | Status | Latency (s) |
|---|---|---|---|---|---|
| task_01 | comparison | 2 | retriever → answer | success | 13.4 |
| task_02 | summarization | 3 | retriever → summarizer → answer | success | 26.1 |
| task_03 | technical-explanation | 2 | retriever → answer | success | 12.8 |
| task_04 | list-generation | 2 | retriever → answer | success | 14.2 |
| task_05 | problem-solution | 3 | retriever → summarizer → answer | success | 27.3 |
| task_06 | comparison | 2 | retriever → answer | success | 13.9 |
| task_07 | technical-explanation | 2 | retriever → answer | success | 12.5 |
| task_08 | evaluation | 2 | retriever → answer | success | 13.1 |
| task_09 | beginner-explanation | 3 | retriever → summarizer → answer | success | 28.4 |
| task_10 | multi-hop | 3 | retriever → retriever → answer | success | 24.7 |

**Summary:**

| Metric | Value |
|---|---|
| Success rate | 10 / 10 (100 %) |
| Average steps per task | 2.4 |
| Average latency | 18.6 s |
| Tasks using summarizer | 3 / 10 |
| Tasks using multi-hop retrieval | 1 / 10 |

---

## 5. Failure Analysis

### 5.1 JSON parsing failures (intermittent)
In ~15% of decision calls, Mistral-7B produces markdown-fenced JSON (e.g., wrapped in ` ```json ``` `) rather than raw JSON. The agent handles this with a stripping step; if parsing still fails, a heuristic fallback selects `retriever` on the first step and `answer` on subsequent steps. This was observed twice across the 10 tasks and did not affect final answer quality.

### 5.2 Tool over-application (task_05)
On the problem-solution task, the agent called `summarizer` on a 200-word passage that did not strictly need summarisation. The summarizer produced a condensed version of similar quality. This indicates the LLM's threshold for "text is too long" is somewhat conservative. A future improvement would add an explicit word-count gate in the tool-selection prompt.

### 5.3 Wrong retrieval query (hypothetical)
If a user asks a highly ambiguous question (e.g., "Tell me about models"), the retriever may surface a mix of documents. The agent mitigated this in task_10 by issuing a second retrieval call with a more specific query. Full recovery mechanisms (iterative query refinement) are listed as a future extension.

### 5.4 No hallucinations observed in agent final answers
Because every final answer generation uses the `RAGPipeline` grounded prompt (*"Answer ONLY using the provided context"*), the agent's answers stayed grounded in retrieved content. The single hallucination observed in Part 1 (Q3) did not recur in agent mode, likely because the summarizer step condensed the context before answer generation.

---

## 6. Model Quality / Latency Trade-offs

| Dimension | Observation |
|---|---|
| **Decision quality** | Mistral-7B correctly selected tools in 9/10 first attempts without the fallback heuristic |
| **Answer quality** | Responses were accurate and well-structured; technical details (formulas, model names) were correctly reproduced from context |
| **Latency** | Each LLM call takes 8–15 s on T4; 3-step tasks take ~27 s end-to-end, which is acceptable for a research/course setting but would require optimisation for production (vLLM, batching) |
| **Memory** | 4-bit quantisation keeps the 7B model at ~4.5 GB VRAM, leaving ~11 GB for ChromaDB and inference overhead on a T4 |
| **Failure modes** | JSON formatting failures are the main reliability concern; a grammar-constrained decoding approach (e.g., Outlines library) would eliminate this |

---

## 7. Conclusions

The agent controller successfully demonstrated autonomous multi-tool coordination using a real open-weight 7B LLM across 10 diverse evaluation tasks. The ReAct decision loop with structured JSON action selection provides full observability — every tool selection, its reasoning, and intermediate outputs are captured in the trace files. The retriever and summarizer tools are complementary: retrieval supplies raw evidence; summarisation distils it; generation synthesises the final grounded answer. The main production limitation is generation latency (~14 s/call on T4), addressable by switching to vLLM or deploying on a faster GPU.
