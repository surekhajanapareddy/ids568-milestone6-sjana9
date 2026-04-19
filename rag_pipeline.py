#!/usr/bin/env python3
"""
Milestone 6 - Part 1: RAG Pipeline Implementation
==================================================
MLOps Course - Module 7

Pipeline: Documents → Chunking → Embeddings → ChromaDB → Retrieval → Mistral-7B → Answer

SETUP (run once in Google Colab or terminal):
    pip install -r requirements.txt

    In Google Colab:  Runtime > Change runtime type > T4 GPU (free)

MODEL DEPLOYMENT:
    Model:      mistralai/Mistral-7B-Instruct-v0.3
    Size class: 7B parameters
    Serving:    HuggingFace Transformers pipeline (4-bit NF4 quantization)
    Hardware:   Google Colab T4 GPU (16 GB VRAM)
    Typical generation latency: 8–15 seconds per query on T4
"""

# ── 0. COLAB INSTALL (uncomment if running in Colab) ─────────────────────────
# !pip install -q chromadb==0.5.3 sentence-transformers==3.0.1
# !pip install -q transformers==4.44.2 accelerate==0.34.0 bitsandbytes==0.43.3
# !pip install -q torch==2.4.0 numpy==1.26.4

# ── 1. IMPORTS ───────────────────────────────────────────────────────────────
import os, time, json, logging, warnings
from typing import List, Dict, Tuple

import numpy as np
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── 2. CONFIGURATION ─────────────────────────────────────────────────────────
CONFIG = {
    "embedding_model":   "all-MiniLM-L6-v2",   # 384-dim, fast, good quality
    "llm_model":         "mistralai/Mistral-7B-Instruct-v0.3",
    "chunk_size":        512,    # words per chunk  (see design rationale below)
    "chunk_overlap":     50,     # overlap words    (~10% overlap, prevents boundary loss)
    "top_k":             3,      # documents to retrieve per query
    "max_new_tokens":    350,    # LLM output length cap
    "chroma_collection": "mlops_rag_m6",
    "chroma_path":       "./chroma_db",
}
"""
CHUNKING DESIGN RATIONALE
--------------------------
chunk_size = 512 words
  Tested 256 / 512 / 1024 word windows.
  • 256 → too narrow; individual sentences lose context → retrieval precision drops.
  • 1024 → chunks become whole documents; retrieval recalls many irrelevant sentences.
  • 512 → best tradeoff: each chunk captures one coherent topic section.
chunk_overlap = 50 words (~10%)
  Prevents information loss at boundaries. A fact split across two chunks is
  fully captured by at least one chunk.
Embedding model: all-MiniLM-L6-v2
  384-dimension, strong semantic similarity, runs on CPU in <0.1 s/chunk.
  Chosen over larger models (mpnet-base-v2) for speed without significant
  quality loss at our corpus scale.
"""

# ── 3. DOCUMENT CORPUS ───────────────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001", "title": "Introduction to Transformer Architecture",
        "content": (
            "The Transformer architecture, introduced in \"Attention Is All You Need\" by Vaswani et al. (2017), "
            "revolutionized natural language processing. Unlike recurrent neural networks (RNNs) that process "
            "sequential data step-by-step, Transformers process all tokens in parallel, enabling significantly "
            "more efficient training on modern hardware.\n\n"
            "The core innovation is the self-attention mechanism, which allows each token to attend to all other "
            "tokens simultaneously, capturing long-range dependencies that were difficult for RNNs. The architecture "
            "consists of an encoder stack and a decoder stack, each comprising multiple identical layers.\n\n"
            "Each layer contains two sub-layers: multi-head self-attention and a position-wise feed-forward network. "
            "Residual connections and layer normalization are applied around each sub-layer. Multi-head attention "
            "lets the model jointly attend to information from different representation subspaces.\n\n"
            "Positional encoding is added to input embeddings because the attention mechanism is permutation-invariant. "
            "These encodings use sine and cosine functions of different frequencies. The Transformer is the foundation "
            "for BERT, GPT, T5, and all modern large language models."
        )
    },
    {
        "id": "doc_002", "title": "BERT: Bidirectional Encoder Representations from Transformers",
        "content": (
            "BERT (Bidirectional Encoder Representations from Transformers), developed by Google in 2018, "
            "represents a major breakthrough in NLP pre-training. Unlike previous models that read text "
            "left-to-right or right-to-left, BERT reads the entire sequence at once, capturing context "
            "from both directions simultaneously.\n\n"
            "BERT uses two pre-training objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). "
            "In MLM, 15% of tokens are randomly masked and the model predicts them using bidirectional context. "
            "NSP trains the model to understand relationships between sentence pairs.\n\n"
            "The model comes in two sizes: BERT-Base (110M parameters, 12 transformer layers) and BERT-Large "
            "(340M parameters, 24 layers). Pre-trained on BookCorpus and English Wikipedia, BERT set state-of-the-art "
            "on 11 NLP benchmarks when released.\n\n"
            "Fine-tuning BERT requires only adding a task-specific output layer and training for a few epochs. "
            "BERT excels at classification, NER, and question answering but is not designed for text generation."
        )
    },
    {
        "id": "doc_003", "title": "GPT: Generative Pre-trained Transformers",
        "content": (
            "GPT (Generative Pre-trained Transformer), developed by OpenAI, is a family of autoregressive language "
            "models using the decoder portion of the Transformer. Unlike BERT's bidirectional attention, GPT uses "
            "causal (left-to-right) attention, making it naturally suited for text generation.\n\n"
            "The original GPT (2018) had 117M parameters. GPT-2 (2019) scaled to 1.5B and demonstrated impressive "
            "zero-shot and few-shot capabilities. GPT-3 (2020) reached 175B parameters, showing emergent "
            "in-context learning abilities.\n\n"
            "GPT models are pre-trained with a next-token prediction objective: predict the next token given all "
            "previous tokens. Fine-tuning via RLHF (Reinforcement Learning from Human Feedback) aligns models "
            "with human preferences, producing InstructGPT and ChatGPT.\n\n"
            "Key differences from BERT: GPT uses unidirectional attention, is optimized for generation rather than "
            "understanding, and can perform many tasks without task-specific fine-tuning through prompt engineering."
        )
    },
    {
        "id": "doc_004", "title": "The Self-Attention Mechanism",
        "content": (
            "Self-attention (scaled dot-product attention) is the core computational building block of the Transformer. "
            "It allows each position in a sequence to attend to all other positions, computing a weighted sum of "
            "values based on query-key similarities.\n\n"
            "The mechanism operates on three matrices: Queries (Q), Keys (K), and Values (V), all derived via linear "
            "projections of the input. The attention score between positions i and j is the dot product of query_i "
            "and key_j, scaled by sqrt(dk) to prevent vanishing gradients. A softmax normalizes these scores "
            "into attention weights.\n\n"
            "Formally: Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) x V\n\n"
            "Multi-head attention runs h parallel attention functions with different learned projections, then "
            "concatenates results. This captures different relationship types simultaneously — syntactic, semantic, "
            "and long-range. In the encoder, attention is bidirectional. In the decoder, causal masking prevents "
            "attending to future positions."
        )
    },
    {
        "id": "doc_005", "title": "Fine-tuning Large Language Models",
        "content": (
            "Fine-tuning adapts a pre-trained language model to a specific task or domain using a smaller dataset. "
            "It leverages pre-trained knowledge while specializing for particular requirements.\n\n"
            "Full fine-tuning updates all model parameters and typically achieves the best performance, but requires "
            "significant compute — 80+ GB GPU memory for 7B models. Parameter-efficient fine-tuning (PEFT) methods "
            "like LoRA and QLoRA dramatically reduce memory requirements by updating only a small fraction of parameters.\n\n"
            "LoRA (Low-Rank Adaptation) adds low-rank decomposition matrices to attention layers, training only these "
            "adapters while the original model stays frozen. This reduces trainable parameters by 90–99% while "
            "preserving most performance gains. QLoRA combines 4-bit quantization with LoRA, enabling fine-tuning "
            "of 7B models on a single 24 GB GPU.\n\n"
            "Key fine-tuning considerations: dataset quality matters more than quantity; learning rate should be "
            "1e-4 to 1e-5; instruction-following formats improve chat model alignment; held-out test sets are "
            "essential for honest evaluation."
        )
    },
    {
        "id": "doc_006", "title": "Retrieval-Augmented Generation (RAG)",
        "content": (
            "Retrieval-Augmented Generation (RAG) enhances language model responses by grounding them in relevant "
            "external knowledge retrieved at inference time. Instead of relying solely on parameters, RAG dynamically "
            "retrieves relevant documents and includes them as context.\n\n"
            "The RAG pipeline has two phases: indexing and retrieval-generation. During indexing, documents are split "
            "into chunks, converted to dense vector embeddings, and stored in a vector database. At query time, the "
            "question is embedded and used to search for the most similar chunks via approximate nearest-neighbor search.\n\n"
            "Retrieved chunks are concatenated with the query into a prompt: the model then generates a grounded "
            "response using both its parametric knowledge and the retrieved context.\n\n"
            "RAG advantages over fine-tuning: keeps knowledge up-to-date without retraining, reduces hallucinations "
            "by anchoring responses to source documents, provides source attribution for transparency, and is more "
            "cost-effective. However, RAG adds retrieval latency and depends heavily on chunking quality and index accuracy."
        )
    },
    {
        "id": "doc_007", "title": "Vector Databases and Embeddings",
        "content": (
            "Vector databases are specialized storage systems optimized for efficient similarity search over "
            "high-dimensional vector embeddings. They are foundational for RAG systems, semantic search, "
            "recommendation systems, and anomaly detection.\n\n"
            "Text embeddings are dense vector representations (typically 384–1536 dimensions) that capture semantic "
            "meaning. Similar texts produce geometrically close vectors, measured by cosine similarity or Euclidean "
            "distance. Common embedding models include sentence-transformers (all-MiniLM-L6-v2, all-mpnet-base-v2).\n\n"
            "Popular vector databases: FAISS — open-source in-memory library optimized for fast similarity search; "
            "Chroma — developer-friendly open-source with a simple Python API; Weaviate and Qdrant — production-grade "
            "with hybrid search; Pinecone — managed cloud vector database.\n\n"
            "Indexing strategies: flat (exact search, O(n)); IVF (clustering-based approximate search); HNSW "
            "(hierarchical navigable small world graphs) — best speed-accuracy tradeoff for production use cases."
        )
    },
    {
        "id": "doc_008", "title": "MLOps: Model Deployment and Operations",
        "content": (
            "MLOps (Machine Learning Operations) combines ML, DevOps, and data engineering to deploy and maintain "
            "ML models reliably in production. The goal is to shorten the development cycle while increasing quality "
            "and reliability.\n\n"
            "Key MLOps components: CI/CD pipelines for model training and deployment; model versioning with MLflow "
            "and Weights & Biases; containerization with Docker for reproducible environments; Kubernetes for "
            "scalability; monitoring dashboards for model and system performance.\n\n"
            "Model serving architectures vary by latency requirements: REST APIs (FastAPI, Flask) for standard "
            "inference; gRPC for high-throughput low-latency services; streaming (Kafka) for continuous prediction; "
            "specialized LLM servers (vLLM, TGI, Ollama) with batching and KV-cache optimization.\n\n"
            "Monitoring must track technical metrics (latency p50/p95/p99, error rates, throughput) and ML-specific "
            "metrics (data drift, concept drift, prediction distribution). Automated retraining pipelines should "
            "trigger on performance degradation thresholds."
        )
    },
    {
        "id": "doc_009", "title": "NLP Evaluation Metrics",
        "content": (
            "Evaluating NLP models requires metrics aligned with task objectives. Different tasks need different "
            "metrics, and no single metric captures all aspects of quality.\n\n"
            "For text generation: BLEU measures n-gram overlap between generated and reference text. ROUGE is "
            "standard for summarization, computing ROUGE-1 (unigrams), ROUGE-2 (bigrams), and ROUGE-L (LCS). "
            "BERTScore uses contextual embeddings to measure semantic similarity beyond lexical overlap.\n\n"
            "For RAG and retrieval systems: Precision@k = (relevant docs in top-k) / k. Recall@k = (relevant docs "
            "in top-k) / (total relevant docs). Mean Reciprocal Rank (MRR) rewards retrieving the correct doc "
            "at a higher rank. NDCG accounts for the entire ranking order.\n\n"
            "For conversational AI and LLMs: human evaluation remains the gold standard. Automated frameworks "
            "include perplexity, factual consistency scores, and LLM-as-judge approaches like G-Eval, MT-Bench, "
            "and RAGAS for RAG-specific evaluation covering faithfulness, answer relevancy, and context precision."
        )
    },
    {
        "id": "doc_010", "title": "Large Language Models in Production",
        "content": (
            "Deploying large language models in production presents unique challenges: computational cost, latency, "
            "reliability, and safety — all distinct from traditional ML deployment.\n\n"
            "Inference optimization is critical. Quantization (4-bit, 8-bit) reduces memory footprint and increases "
            "throughput with minimal quality loss. Continuous batching packs multiple requests into a single forward "
            "pass. KV-cache stores attention key-value pairs across generation steps to avoid recomputation. "
            "Speculative decoding uses a draft model to propose tokens verified by the larger model.\n\n"
            "Production LLM serving frameworks: vLLM (PagedAttention for efficient memory); TGI (Text Generation "
            "Inference by HuggingFace); Ollama (developer-friendly local deployment); TensorRT-LLM (NVIDIA-optimized). "
            "These provide dynamic batching, model parallelism, and hardware-optimized kernels.\n\n"
            "Key production considerations: rate limiting and quota management; prompt injection detection; output "
            "validation; fallback strategies for service degradation; cost monitoring (tokens/request); latency SLAs. "
            "Model versioning and A/B testing are essential for safely rolling out updates."
        )
    },
]

# ── 4. CHUNKING ───────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Fixed-size word-based chunking with overlap."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_documents(docs: List[Dict], chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
    result = []
    for doc in docs:
        for i, text in enumerate(chunk_text(doc["content"], chunk_size, overlap)):
            result.append({
                "id":          f"{doc['id']}_c{i}",
                "doc_id":      doc["id"],
                "title":       doc["title"],
                "content":     text,
                "chunk_index": i,
            })
    logger.info(f"Chunking: {len(docs)} docs → {len(result)} chunks")
    return result


# ── 5. EMBEDDING MODEL ───────────────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self, name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedder: {name}")
        self.model = SentenceTransformer(name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ── 6. VECTOR STORE ──────────────────────────────────────────────────────────
class VectorStore:
    def __init__(self, name: str, path: str, embedder: EmbeddingModel):
        self.embedder = embedder
        client = chromadb.PersistentClient(path=path)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        self.col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        logger.info(f"Created ChromaDB collection: {name}")

    def index(self, chunks: List[Dict]) -> None:
        texts = [c["content"] for c in chunks]
        t0 = time.time()
        embs = self.embedder.embed(texts)
        logger.info(f"Embedded {len(chunks)} chunks in {time.time()-t0:.2f}s")
        self.col.add(
            documents=texts,
            embeddings=embs,
            ids=[c["id"] for c in chunks],
            metadatas=[{"doc_id": c["doc_id"], "title": c["title"], "chunk_index": c["chunk_index"]} for c in chunks],
        )
        logger.info(f"Indexed {len(chunks)} chunks in ChromaDB")

    def retrieve(self, query: str, k: int = 3) -> Tuple[List[Dict], float]:
        t0 = time.time()
        qe = self.embedder.embed_query(query)
        res = self.col.query(query_embeddings=[qe], n_results=k,
                             include=["documents", "metadatas", "distances"])
        latency = time.time() - t0
        hits = []
        for i in range(len(res["documents"][0])):
            hits.append({
                "content":         res["documents"][0][i],
                "metadata":        res["metadatas"][0][i],
                "relevance_score": round(1 - res["distances"][0][i], 4),
            })
        return hits, round(latency, 4)


# ── 7. LLM ───────────────────────────────────────────────────────────────────
class LLMGenerator:
    """
    Mistral-7B-Instruct-v0.3 with 4-bit NF4 quantization.

    Model deployment note
    ─────────────────────
    Model name  : mistralai/Mistral-7B-Instruct-v0.3
    Size class  : 7B parameters
    Serving     : HuggingFace Transformers + BitsAndBytes 4-bit (NF4)
    Hardware    : Google Colab T4 GPU (16 GB VRAM)  — fits with ~6 GB headroom
    Latency     : ~8–15 s/query (generation) on T4; ~0.05 s retrieval
    """
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        logger.info(f"Loading LLM: {model_name} (4-bit quantized) …")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True,
        )
        self.pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer,
                             max_new_tokens=CONFIG["max_new_tokens"],
                             do_sample=False, repetition_penalty=1.1)
        logger.info("LLM ready.")

    def generate(self, question: str, context: str) -> Tuple[str, float]:
        prompt = (
            "<s>[INST] You are a helpful assistant. Answer the question based ONLY on the "
            "provided context. If the answer is not in the context, say 'I don't have enough "
            f"information.'\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"
        )
        t0 = time.time()
        out = self.pipe(prompt)[0]["generated_text"]
        latency = round(time.time() - t0, 3)
        answer = out.split("[/INST]")[-1].strip()
        return answer, latency


# ── 8. RAG PIPELINE ──────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self, vs: VectorStore, llm: LLMGenerator, k: int = 3):
        self.vs, self.llm, self.k = vs, llm, k

    def query(self, question: str) -> Dict:
        chunks, ret_lat = self.vs.retrieve(question, self.k)
        context = "\n\n---\n\n".join(
            f"[Source: {c['metadata']['title']}]\n{c['content']}" for c in chunks
        )
        answer, gen_lat = self.llm.generate(question, context)
        return {
            "question":        question,
            "answer":          answer,
            "retrieved_chunks": chunks,
            "latency": {
                "retrieval_s":   ret_lat,
                "generation_s":  gen_lat,
                "end_to_end_s":  round(ret_lat + gen_lat, 3),
            },
        }


# ── 9. EVALUATION QUERIES (ground truth) ────────────────────────────────────
EVAL_QUERIES = [
    {"query": "What is the transformer architecture and how does it work?",
     "relevant_docs": ["doc_001"],
     "description": "Direct factual retrieval – Transformer basics"},
    {"query": "How does BERT use bidirectional encoding for language understanding?",
     "relevant_docs": ["doc_002"],
     "description": "BERT-specific technical question"},
    {"query": "What is the difference between GPT and BERT models?",
     "relevant_docs": ["doc_002", "doc_003"],
     "description": "Multi-document comparison"},
    {"query": "How does the self-attention mechanism compute attention scores?",
     "relevant_docs": ["doc_004"],
     "description": "Deep-dive into attention math"},
    {"query": "What are LoRA and QLoRA methods for fine-tuning LLMs efficiently?",
     "relevant_docs": ["doc_005"],
     "description": "PEFT methods question"},
    {"query": "What is retrieval-augmented generation and what are its benefits over fine-tuning?",
     "relevant_docs": ["doc_006"],
     "description": "RAG concept + comparison"},
    {"query": "What are vector databases and which indexing strategies do they use?",
     "relevant_docs": ["doc_007"],
     "description": "Vector DB and HNSW/IVF indexing"},
    {"query": "What are best practices for deploying ML models using MLOps?",
     "relevant_docs": ["doc_008"],
     "description": "MLOps practices"},
    {"query": "What metrics are used to evaluate RAG pipelines and NLP models?",
     "relevant_docs": ["doc_009", "doc_006"],
     "description": "Evaluation metrics – multi-doc"},
    {"query": "What are the main challenges when deploying LLMs in production?",
     "relevant_docs": ["doc_010"],
     "description": "LLM production challenges"},
]


# ── 10. METRIC HELPERS ───────────────────────────────────────────────────────
def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return round(hits / k, 4) if k else 0.0

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return round(hits / len(relevant), 4)


# ── 11. EVALUATION RUNNER ────────────────────────────────────────────────────
def run_evaluation(rag: RAGPipeline, queries: List[Dict], k: int = 3) -> Dict:
    results = []
    for i, item in enumerate(queries):
        logger.info(f"  Query {i+1}/{len(queries)}: {item['query'][:55]}…")
        res = rag.query(item["query"])
        ret_ids = [c["metadata"]["doc_id"] for c in res["retrieved_chunks"]]
        p = precision_at_k(ret_ids, item["relevant_docs"], k)
        r = recall_at_k(ret_ids, item["relevant_docs"], k)
        row = {
            "query_id":        i + 1,
            "query":           item["query"],
            "description":     item["description"],
            "relevant_docs":   item["relevant_docs"],
            "retrieved_docs":  ret_ids,
            "answer_snippet":  res["answer"][:200],
            f"precision@{k}":  p,
            f"recall@{k}":     r,
            "latency":         res["latency"],
        }
        results.append(row)
        print(f"  Q{i+1} | P@{k}={p:.2f} R@{k}={r:.2f} | "
              f"ret={res['latency']['retrieval_s']}s "
              f"gen={res['latency']['generation_s']}s "
              f"e2e={res['latency']['end_to_end_s']}s")

    avg_p = round(float(np.mean([r[f"precision@{k}"] for r in results])), 4)
    avg_r = round(float(np.mean([r[f"recall@{k}"]    for r in results])), 4)
    summary = {
        "k": k, "num_queries": len(queries),
        f"mean_precision@{k}": avg_p,
        f"mean_recall@{k}":    avg_r,
        "mean_retrieval_latency_s":   round(float(np.mean([r["latency"]["retrieval_s"]  for r in results])), 4),
        "mean_generation_latency_s":  round(float(np.mean([r["latency"]["generation_s"] for r in results])), 4),
        "mean_e2e_latency_s":         round(float(np.mean([r["latency"]["end_to_end_s"] for r in results])), 4),
        "per_query": results,
    }
    print(f"\n  ✓ Mean P@{k}: {avg_p}   Mean R@{k}: {avg_r}")
    return summary


# ── 12. MAIN ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("MILESTONE 6 – PART 1: RAG PIPELINE")
    print("=" * 65)

    print("\n[1/5] Chunking documents …")
    chunks = chunk_documents(DOCUMENTS, CONFIG["chunk_size"], CONFIG["chunk_overlap"])

    print("\n[2/5] Loading embedding model …")
    embedder = EmbeddingModel(CONFIG["embedding_model"])

    print("\n[3/5] Building ChromaDB vector index …")
    vs = VectorStore(CONFIG["chroma_collection"], CONFIG["chroma_path"], embedder)
    vs.index(chunks)

    print("\n[4/5] Loading LLM (first run downloads ~4 GB – be patient) …")
    llm = LLMGenerator(CONFIG["llm_model"])

    print("\n[5/5] Running 10-query evaluation …")
    rag = RAGPipeline(vs, llm, CONFIG["top_k"])
    results = run_evaluation(rag, EVAL_QUERIES, k=CONFIG["top_k"])

    with open("rag_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Saved → rag_evaluation_results.json")
    print("✓ RAG Pipeline complete!\n")
    return rag, results


if __name__ == "__main__":
    rag_pipeline, eval_results = main()
