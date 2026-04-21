# Milestone 6: RAG Pipeline & Agentic System
## MLOps Course – Module 7

---

## Overview

This repository implements two components:

1. **Part 1 – RAG Pipeline** (`rag_pipeline.py`): A complete Retrieval-Augmented Generation system using ChromaDB, sentence-transformers, and Mistral-7B-Instruct.
2. **Part 2 – Agent Controller** (`agent_controller.py`): A ReAct-style multi-tool agent that autonomously selects between a retriever and summarizer to answer complex tasks.

---

## Repository Structure

```
ids568-milestone6-[netid]/
├── rag_pipeline.py              # Part 1: RAG implementation
├── agent_controller.py          # Part 2: Agent controller
├── rag_evaluation_report.md     # Part 1: Evaluation report (~2 pages)
├── rag_pipeline_diagram.md      # Part 1: Pipeline architecture diagram
├── agent_report.md              # Part 2: Agent analysis report
├── agent_traces/                # Part 2: 10 evaluation task traces
│   ├── task_01.json … task_10.json
│   └── _summary.json            # (generated after running agent)
├── requirements.txt             # Pinned dependencies
└── README.md                    # This file
```

---

## Model Deployment Details

| Field | Value |
|---|---|
| **Model name** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Size class** | 7B parameters |
| **Serving stack** | HuggingFace Transformers + BitsAndBytes 4-bit NF4 quantization |
| **Hardware** | Google Colab T4 GPU (16 GB VRAM) |
| **VRAM footprint** | ~4.5 GB (4-bit quantized) |
| **Typical generation latency** | 8–15 seconds per query on T4 |
| **Model download** | Auto-downloaded from HuggingFace Hub on first run (~4 GB) |

---

## Setup Instructions

### Option A — Google Colab (Recommended for beginners)

1. Open [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**: `Runtime > Change runtime type > T4 GPU`
3. Upload all files to Colab or clone the repository:
   ```python
   !git clone https://github.com/[your-username]/ids568-milestone6-[sjan9].git
   %cd ids568-milestone6-[sjana9]
   ```
4. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
5. Run the RAG pipeline (Step 4 downloads Mistral-7B on first run — ~4 GB):
   ```bash
   !python rag_pipeline.py
   ```

### Option B — Local Machine (requires NVIDIA GPU with 8+ GB VRAM)

```bash
# 1. Clone the repository
git clone https://github.com/[your-username]/ids568-milestone6-[sjana9].git
cd ids568-milestone6-[sjana9]

# 2. Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Part 1 (RAG pipeline)
python rag_pipeline.py

# 5. Run Part 2 (Agent controller) — requires Part 1 completed first
python agent_controller.py
```

---

## Usage Examples

### Part 1 — RAG Pipeline

```python
from rag_pipeline import (
    DOCUMENTS, CONFIG,
    chunk_documents, EmbeddingModel, VectorStore, LLMGenerator, RAGPipeline
)

# Build the pipeline
chunks   = chunk_documents(DOCUMENTS, CONFIG["chunk_size"], CONFIG["chunk_overlap"])
embedder = EmbeddingModel(CONFIG["embedding_model"])
vs       = VectorStore(CONFIG["chroma_collection"], CONFIG["chroma_path"], embedder)
vs.index(chunks)
llm      = LLMGenerator(CONFIG["llm_model"])
rag      = RAGPipeline(vs, llm, k=3)

# Ask a question
result = rag.query("What is the self-attention mechanism?")
print(result["answer"])
print("Latency:", result["latency"])
```

### Part 2 — Agent Controller

```python
from agent_controller import build_agent, AGENT_TASKS, evaluate_agent

# Build the agent (initialises all components)
agent = build_agent()

# Run a single task
trace = agent.run("What is RAG and how does it differ from fine-tuning?")
print(trace["final_answer"])
print("Steps taken:", [s["action"] for s in trace["steps"]])

# Run all 10 evaluation tasks and save traces
evaluate_agent(agent, AGENT_TASKS, output_dir="agent_traces")
```

---

## Architecture Overview

### RAG Pipeline

```
Documents → Chunker (512w, 50w overlap)
         → Embedder (all-MiniLM-L6-v2, 384-dim)
         → ChromaDB (HNSW cosine index)
                         ↑
User Query → Query Embedder → Top-3 Retrieval
                                    ↓
                           Context Formatter
                                    ↓
                        Mistral-7B-Instruct (4-bit)
                                    ↓
                           Grounded Answer
```

### Agent Controller

```
Task → LLM Decision (Mistral-7B) → {retriever | summarizer | answer}
           ↑                              ↓
     Action history              Tool executed
           ↑                              ↓
     Result appended ←──────── Structured result
                   (loop until action == "answer")
```

---

## Known Limitations

1. **Generation latency**: ~11 s/query on Colab T4. For production, switch to vLLM with continuous batching (2–4× speedup).
2. **Small corpus**: The 10-document corpus is for demonstration. Production RAG systems index thousands of documents — ChromaDB scales well, but the embedding step becomes the bottleneck.
3. **JSON parsing**: Mistral-7B occasionally wraps its JSON decision in markdown code fences. The agent handles this with a stripping step and a heuristic fallback.
4. **No query expansion**: Ambiguous queries may retrieve topically-adjacent but not perfectly relevant documents. A reranker or query expansion step would improve precision.
5. **Single-GPU only**: The current implementation loads the model on one GPU. Multi-GPU or CPU-only inference is not configured.
6. **HuggingFace token**: Some gated models (e.g., Llama-3) require a HuggingFace account token. Mistral-7B-Instruct-v0.3 is public and does not require a token.

---

## Running the Automated Sanity Checks

From the repository root, run:

```bash
# Check all required files exist
for file in rag_pipeline.py agent_controller.py rag_evaluation_report.md \
            rag_pipeline_diagram.md agent_report.md requirements.txt README.md; do
    [ -f "$file" ] && echo "✓ $file" || echo "✗ $file MISSING"
done

# Check agent traces
ls -1 agent_traces/*.json 2>/dev/null | wc -l  # should be >= 10
```
