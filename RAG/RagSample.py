# Minimal RAG: embeddings + similarity search + prompt to a small LLM
# ---------------------------------------------------------------
# What it does:
# 1) Chunk & embed docs with SentenceTransformers
# 2) Retrieve top-k using cosine similarity (sklearn NearestNeighbors)
# 3) Build a simple RAG prompt and generate with FLAN-T5

from typing import List, Tuple
from dataclasses import dataclass
import textwrap

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Sample corpus (tiny on purpose)
# -----------------------------

DOCS = [
    ("doc1", """A savings account is a deposit account that earns interest over time.
Banks use these deposits to fund loans, while customers benefit from liquidity and small returns."""),
    ("doc2", """Credit cards allow customers to borrow money for purchases and pay later.
They often come with interest rates, credit limits, and reward programs."""),
    ("doc3", """Machine learning models can be applied in fraud detection by identifying unusual transaction patterns.
Isolation Forest and neural networks are common algorithms used in this space."""),
    ("doc4", """Blockchain is a decentralized ledger technology that records transactions across a peer-to-peer network.
It ensures transparency, immutability, and security without central authority."""),
]

# -----------------------------
# Simple chunking (one chunk per doc for brevity)
# In real systems, you'd split by ~300-800 tokens with overlap.
# -----------------------------
@dataclass
class Chunk:
    id: str
    text: str

def make_chunks(docs: List[Tuple[str, str]]) -> List[Chunk]:
    return [Chunk(id=doc_id, text=txt.strip()) for doc_id, txt in docs]

# -----------------------------
# Embedder
# -----------------------------
class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts: List[str]) -> np.ndarray:
        # returns float32 matrix [n, d]
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype(np.float32)

# -----------------------------
# Retriever using cosine similarity via NearestNeighbors(metric='cosine')
# -----------------------------
class Retriever:
    def __init__(self, embeddings: np.ndarray):
        # sklearn's cosine distance = 1 - cosine_similarity
        self.nn = NearestNeighbors(n_neighbors=3, metric="cosine")
        self.nn.fit(embeddings)
        self.embeddings = embeddings

    def top_k(self, query_emb: np.ndarray, k: int = 3):
        distances, idx = self.nn.kneighbors(query_emb.reshape(1, -1), n_neighbors=k)
        # Convert cosine distance -> similarity
        sims = 1.0 - distances[0]
        return list(zip(idx[0].tolist(), sims.tolist()))

# -----------------------------
# Tiny generator (FLAN-T5)
# -----------------------------
class Generator:
    def __init__(self, model_name: str = "google/flan-t5-small", max_new_tokens: int = 180):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Prompt builder
# -----------------------------
def build_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts))
    instruction = (
        "You are a helpful assistant. Using ONLY the provided context, answer the question concisely. "
        "If the answer isn't in the context, say you don't know.\n"
    )
    prompt = f"{instruction}\n{context_block}\n\n[Question]\n{query}\n\n[Answer]"
    return textwrap.dedent(prompt).strip()

# -----------------------------
# Put it all together
# -----------------------------
def main():
    # 1) Chunks
    chunks = make_chunks(DOCS)
    texts = [c.text for c in chunks]

    # 2) Embeddings
    emb = Embedder()
    chunk_vecs = emb.encode(texts)

    # 3) Retriever
    retriever = Retriever(chunk_vecs)

    # Example queries
    queries = [
       "How do banks use savings accounts?",

"What is the role of machine learning in fraud detection?",

"Why is blockchain considered secure?",

"What are the benefits of credit cards?"
    ]

    gen = Generator()

    for q in queries:
        q_vec = emb.encode([q])[0]
        top = retriever.top_k(q_vec, k=2)  # get top-2

        # Get the retrieved contexts
        ctx_texts = [chunks[i].text for i, _sim in top]

        # Build RAG prompt
        prompt = build_prompt(q, ctx_texts)

        print("="*80)
        print("Query:", q)
        print("- Retrieved contexts (top-2):")
        for rank, (idx, sim) in enumerate(top, start=1):
            print(f"  {rank}. chunk={chunks[idx].id}  sim={sim:.3f}")

        # 4) Generate
        answer = gen.generate(prompt)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()

