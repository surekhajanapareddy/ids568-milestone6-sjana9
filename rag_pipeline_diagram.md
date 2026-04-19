# RAG Pipeline Diagram
## Milestone 6 – Part 1 | MLOps Course – Module 7

---

## System Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║                         RAG PIPELINE                                 ║
║                   (Milestone 6 – Part 1)                             ║
╚══════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────┐
  │                     INDEXING PHASE (offline)                    │
  └─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  Raw         │   10 plain-text documents
  │  Documents   │   (ML topics, ~300 words each)
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  CHUNKER                                                     │
  │  ─────────────────────────────────────────────────────────── │
  │  Strategy : Fixed-size word windows                          │
  │  chunk_size = 512 words                                      │
  │  chunk_overlap = 50 words (~10 %)                            │
  │  Output   : List[{id, doc_id, title, content, chunk_index}]  │
  └──────────────────────┬───────────────────────────────────────┘
                         │  ~10–15 chunks per document
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  EMBEDDER                                                    │
  │  ─────────────────────────────────────────────────────────── │
  │  Model    : all-MiniLM-L6-v2 (sentence-transformers)         │
  │  Dims     : 384 floating-point values per chunk              │
  │  Latency  : ~0.01 s / chunk on CPU                           │
  │  Output   : List[float[384]]                                 │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  VECTOR STORE  (ChromaDB – persistent)                       │
  │  ─────────────────────────────────────────────────────────── │
  │  Index    : HNSW (cosine similarity)                         │
  │  Stores   : {id, embedding[384], text, metadata}             │
  │  Location : ./chroma_db/  (on disk)                          │
  └──────────────────────────────────────────────────────────────┘


  ┌─────────────────────────────────────────────────────────────────┐
  │                  RETRIEVAL + GENERATION PHASE (online)          │
  └─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  User        │
  │  Question    │  e.g. "What is self-attention?"
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  QUERY EMBEDDER                                              │
  │  Same all-MiniLM-L6-v2 model                                 │
  │  Encodes question → float[384]                               │
  │  Latency : ~0.005 s                                          │
  └──────────────────────┬───────────────────────────────────────┘
                         │  query vector
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  RETRIEVER                                                   │
  │  ─────────────────────────────────────────────────────────── │
  │  ChromaDB HNSW approximate nearest-neighbour search          │
  │  Returns top-k=3 chunks by cosine similarity                 │
  │  Latency : ~0.04–0.07 s                                      │
  │  Output  : List[{content, title, doc_id, relevance_score}]   │
  └──────────────────────┬───────────────────────────────────────┘
                         │  3 relevant passages
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  CONTEXT FORMATTER                                           │
  │  ─────────────────────────────────────────────────────────── │
  │  Concatenates retrieved passages with source labels:         │
  │                                                              │
  │  "[Source: Title A]\n<chunk text>\n---\n                     │
  │   [Source: Title B]\n<chunk text>\n---\n..."                 │
  │                                                              │
  │  Decision point: if no chunks retrieved → return "no info"   │
  └──────────────────────┬───────────────────────────────────────┘
                         │  formatted context string
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  PROMPT BUILDER                                              │
  │  ─────────────────────────────────────────────────────────── │
  │  Template:                                                   │
  │  <s>[INST] Answer ONLY using the provided context.           │
  │  Context: {formatted_context}                                │
  │  Question: {user_question} [/INST]                           │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  GENERATOR  (LLM)                                            │
  │  ─────────────────────────────────────────────────────────── │
  │  Model   : mistralai/Mistral-7B-Instruct-v0.3                │
  │  Quant   : 4-bit NF4 (BitsAndBytes)                          │
  │  Serving : HuggingFace Transformers pipeline                 │
  │  HW      : Google Colab T4 GPU (16 GB VRAM)                  │
  │  Latency : 8–15 s / query                                    │
  │  max_new_tokens = 350                                        │
  │                                                              │
  │  Decision point: if context is empty → "no info" response    │
  └──────────────────────┬───────────────────────────────────────┘
                         │  generated answer text
                         ▼
  ┌──────────────┐
  │  Final       │  Grounded answer + latency metrics
  │  Answer      │  + retrieved sources (for attribution)
  └──────────────┘
```

---

## Data Flow Summary

```
Documents → Chunker → Embedder → ChromaDB (HNSW)
                                       ↑
User Query → Query Embedder ──────────┘
                                       ↓
                              Top-3 Chunks Retrieved
                                       ↓
                              Context Formatter
                                       ↓
                              Prompt Builder
                                       ↓
                         Mistral-7B-Instruct (4-bit)
                                       ↓
                              Grounded Answer
```

---

## Component Summary

| Component | Technology | Role |
|---|---|---|
| Chunker | Custom Python (word-window) | Splits documents into 512-word overlapping chunks |
| Embedder | `all-MiniLM-L6-v2` | Converts text → 384-dim semantic vectors |
| Vector Store | ChromaDB (HNSW, cosine) | Stores and indexes chunk embeddings |
| Retriever | ChromaDB `.query()` | Finds top-k most similar chunks for a query |
| Context Formatter | Python string concat | Assembles prompt context from retrieved chunks |
| Generator | Mistral-7B-Instruct-v0.3 (4-bit) | Produces grounded natural-language answers |

---

## Decision Points

1. **Chunk boundary** — if a fact spans two chunks, the 50-word overlap ensures it appears complete in at least one chunk.
2. **Retrieval fallback** — if relevance score of top-1 chunk < 0.2, the pipeline returns *"I don't have enough information"* rather than hallucinating.
3. **Context length guard** — if concatenated context exceeds 3000 words, only top-2 chunks are used to stay within the model's context window.
4. **Empty generation guard** — if the LLM output after `[/INST]` is empty or shorter than 10 characters, a retry is triggered with a simplified prompt.
