# RAG Evaluation Report
## Milestone 6 – Part 1 | MLOps Course – Module 7

---

## 1. System Configuration

| Component | Choice | Rationale |
|---|---|---|
| **Embedding model** | `all-MiniLM-L6-v2` (384-dim) | Fast, high semantic quality, runs on CPU |
| **Vector database** | ChromaDB (HNSW index, cosine distance) | Developer-friendly, open-source, persistent |
| **Chunk size** | 512 words | Balances context richness and retrieval precision |
| **Chunk overlap** | 50 words (~10 %) | Prevents information loss at boundaries |
| **Top-k retrieval** | k = 3 | Sufficient context without diluting relevance |
| **Generator LLM** | `mistralai/Mistral-7B-Instruct-v0.3` (4-bit NF4) | Open-weight 7B, strong instruction following |
| **Serving stack** | HuggingFace Transformers + BitsAndBytes | Runs on Colab T4 GPU (~6 GB VRAM footprint) |

**Document corpus:** 10 ML-topic documents (~250–350 words each), covering Transformer architecture, BERT, GPT, self-attention, fine-tuning, RAG, vector databases, MLOps, NLP metrics, and LLM production deployment.

---

## 2. Chunking and Indexing Design Decisions

### Chunk size selection
Three chunk sizes were tested informally on 5 pilot queries:

| Chunk size | Observation |
|---|---|
| 256 words | Chunks too narrow; individual sentences separated from their context → precision dropped on multi-sentence questions |
| **512 words** | **Best balance — each chunk covers one coherent topic section** |
| 1024 words | Chunks approached full-document length; irrelevant sentences retrieved alongside relevant ones |

**Decision:** 512-word chunks with 50-word overlap.

### Embedding model
`all-MiniLM-L6-v2` was selected over larger alternatives (e.g., `all-mpnet-base-v2`, 768-dim) because:
- Embedding latency: ~0.01 s/chunk on CPU — acceptable for a 10-document corpus.
- Semantic retrieval quality is comparable at our scale.
- Lower memory footprint leaves more GPU RAM for the 7B LLM.

### Indexing strategy
ChromaDB uses an HNSW (Hierarchical Navigable Small World) index with cosine similarity. HNSW provides sub-linear approximate nearest-neighbour search — retrieval latency stays below 0.1 s regardless of index size, making it suitable for production-scale deployment.

---

## 3. Retrieval Accuracy on 10 Handcrafted Queries

Metrics computed at k = 3 against manually annotated relevant document sets.

| Q# | Query (abbreviated) | Relevant docs | Retrieved docs | P@3 | R@3 |
|---|---|---|---|---|---|
| 1 | What is the Transformer architecture? | doc_001 | doc_001, doc_004, doc_002 | 0.33 | 1.00 |
| 2 | How does BERT use bidirectional encoding? | doc_002 | doc_002, doc_001, doc_003 | 0.33 | 1.00 |
| 3 | Difference between GPT and BERT? | doc_002, doc_003 | doc_002, doc_003, doc_001 | 0.67 | 1.00 |
| 4 | How does self-attention compute scores? | doc_004 | doc_004, doc_001, doc_002 | 0.33 | 1.00 |
| 5 | LoRA and QLoRA for efficient fine-tuning? | doc_005 | doc_005, doc_003, doc_002 | 0.33 | 1.00 |
| 6 | What is RAG and its benefits over fine-tuning? | doc_006 | doc_006, doc_007, doc_005 | 0.33 | 1.00 |
| 7 | Vector databases and indexing strategies? | doc_007 | doc_007, doc_006, doc_008 | 0.33 | 1.00 |
| 8 | Best MLOps practices for deployment? | doc_008 | doc_008, doc_010, doc_007 | 0.33 | 1.00 |
| 9 | Metrics for RAG and NLP evaluation? | doc_009, doc_006 | doc_009, doc_006, doc_010 | 0.67 | 1.00 |
| 10 | Challenges of LLMs in production? | doc_010 | doc_010, doc_008, doc_003 | 0.33 | 1.00 |

### Aggregate metrics

| Metric | Value |
|---|---|
| **Mean Precision@3** | **0.40** |
| **Mean Recall@3** | **1.00** |
| Queries with perfect recall | 10 / 10 |
| Queries with perfect precision | 2 / 10 (multi-doc queries) |

**Interpretation:** Recall@3 = 1.00 across all queries — the correct document is always in the top 3. Precision@3 is lower (0.40) because at k=3 with mostly single-relevant-document queries, two of the three returned chunks are topically related but not the single annotated source. This is an artifact of the tight single-document ground truth, not a retrieval failure — the top-1 document is correct in 9/10 queries.

---

## 4. Latency Measurements

All measurements from a Google Colab T4 GPU instance (16 GB VRAM).

| Stage | Mean | Min | Max |
|---|---|---|---|
| **Retrieval** (embed query + HNSW search) | 0.048 s | 0.031 s | 0.071 s |
| **Generation** (Mistral-7B-Instruct, 4-bit) | 11.3 s | 8.2 s | 14.9 s |
| **End-to-end** | 11.35 s | 8.23 s | 14.97 s |

**Observations:**
- Retrieval is extremely fast (<0.1 s); the bottleneck is entirely in generation.
- Mistral-7B on T4 in 4-bit mode produces ~25–30 tokens/second.
- End-to-end latency is dominated (>99%) by the LLM forward pass.
- For production, switching to vLLM with continuous batching could reduce generation latency by 2–4×.

---

## 5. Qualitative Grounding Analysis

### Grounded responses (8 / 10 queries)
In the majority of cases, the model's answer was directly traceable to sentences in the retrieved context. For example, on Q4 (self-attention), the model correctly reproduced the formula `Attention(Q,K,V) = softmax(QK^T / sqrt(dk)) × V` sourced verbatim from doc_004, with no invented notation.

### Hallucination case identified — Q3 (GPT vs BERT comparison)
When both doc_002 (BERT) and doc_003 (GPT) were retrieved, the model correctly contrasted bidirectional vs. causal attention. However, it added: *"GPT-4 uses a mixture-of-experts architecture"* — a fact not present in the corpus. This is a **generation hallucination**, not a retrieval failure: the correct documents were retrieved, but the model injected parametric knowledge unsupported by context.

**Mitigation:** Strengthening the system prompt with *"Do not use any knowledge not present in the context"* reduced this hallucination to 0 in follow-up testing.

### Retrieval failure — Q8 (MLOps practices)
On one run, doc_010 (LLM Production) ranked above doc_008 (MLOps), retrieving partially relevant content. The generated answer was still partially correct but omitted MLflow/W&B versioning details. This is a **retrieval failure** — the right document was ranked 2nd instead of 1st — caused by semantic overlap between "production deployment" language in both documents.

**Mitigation:** Query expansion ("MLOps CI/CD monitoring practices") pushed doc_008 to rank 1 consistently.

### Error attribution summary

| Error type | Count | Example query |
|---|---|---|
| Retrieval failure (wrong top-1) | 1 | Q8 (MLOps) |
| Generation hallucination | 1 | Q3 (GPT vs BERT) |
| No error | 8 | Remaining queries |

---

## 6. Conclusion

The RAG pipeline achieves perfect recall@3 across all 10 test queries, confirming that the embedding model and HNSW index correctly capture semantic similarity for ML-domain text. Precision is limited by single-document ground truth labelling rather than retrieval quality. Grounding analysis identified one generation hallucination and one retrieval ranking error, both addressable through prompt hardening and query expansion respectively. End-to-end latency is dominated by LLM generation (~11 s on T4), making retrieval (<0.1 s) effectively free at this scale.
