"""
Core AI components for Prometheus AI Orchestrator
Contains quantum-inspired cognitive architecture and neuromorphic memory systems
"""

import logging
from typing import List, Optional, Dict, Any

# Try to import optional dependencies
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

# Try to import FAISS, fall back gracefully if not available
try:
    import faiss  # Optional; required for NeuromorphicMemory
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)


class QuantumCognitiveCore:
    """Enhanced quantum-inspired neural network with improved entanglement and superposition simulation for superior decision making."""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_qubits=8
    ):  # Increased qubits for better parallelism
        if torch is None or nn is None:
            raise ImportError("PyTorch not available for QuantumCognitiveCore")

        super(QuantumCognitiveCore, self).__init__()
        self.num_qubits = num_qubits
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Enhanced quantum-inspired layers with variational quantum circuits simulation
        self.quantum_encoder = nn.Linear(input_dim, num_qubits * hidden_dim)
        self.quantum_circuit = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  # Added normalization for stability
                    nn.ReLU(),
                )
                for _ in range(num_qubits)
            ]
        )
        self.quantum_decoder = nn.Linear(num_qubits * hidden_dim, output_dim)

        # Improved entanglement with learnable Hadamard gates simulation
        self.entanglement = nn.Parameter(torch.randn(num_qubits, num_qubits))
        self.superposition_weights = nn.Parameter(
            torch.ones(num_qubits)
        )  # For superposition collapse

        # Enhanced cognitive state with memory gates
        self.cognitive_state = torch.zeros(1, hidden_dim)
        # Use output_dim for attention so it matches decoder output; keeps dimensions consistent
        self.attention_weights = nn.MultiheadAttention(
            output_dim, num_heads=8, batch_first=True
        )
        self.memory_gate = nn.Linear(output_dim, 1)  # For gating long-term memory

    def forward(self, x, cognitive_state=None):
        # Encode input into quantum superposition state
        encoded = torch.tanh(self.quantum_encoder(x))
        batch_size = x.size(0)
        encoded = encoded.view(batch_size, self.num_qubits, self.hidden_dim)

        # Apply variational quantum circuit transformations
        quantum_states = []
        for i in range(self.num_qubits):
            state = self.quantum_circuit[i](encoded[:, i, :])
            state = torch.sigmoid(state)  # Activation for qubit states
            quantum_states.append(state)

        # Simulate entanglement and superposition collapse
        entangled = torch.stack(quantum_states, dim=1)  # (B, Q, H)
        # Build entanglement matrix over qubits and apply across qubit axis
        ent_matrix = self.entanglement @ torch.diag(
            self.superposition_weights
        )  # (Q, Q)
        entangled = torch.einsum("bqh,qq->bqh", entangled, ent_matrix)  # (B, Q, H)
        entangled = entangled.reshape(batch_size, -1)

        # Decode to classical output with noise for exploration
        noise = torch.randn_like(entangled) * 0.01  # Quantum noise simulation
        output = self.quantum_decoder(entangled + noise)

        # Enhanced cognitive state update with memory gating
        if cognitive_state is not None:
            # Ensure cognitive_state shape is (batch, seq_len=1, embed=output_dim)
            if cognitive_state.dim() == 2:
                cognitive_state = cognitive_state.unsqueeze(1)
            # Project query over current output context
            attn_output, _ = self.attention_weights(
                cognitive_state, output.unsqueeze(1), output.unsqueeze(1)
            )
            # attn_output: (batch, 1, output_dim)
            gate = torch.sigmoid(self.memory_gate(attn_output.squeeze(1)))
            new_cognitive_state = cognitive_state.squeeze(
                1
            ) * gate + attn_output.squeeze(1) * (1 - gate)
            return output, new_cognitive_state

        return output


class NeuromorphicMemory:
    """Advanced memory system with improved clustering, lazy loading, and active forgetting mechanism."""

    def __init__(
        self, memory_dim=768, num_clusters=20
    ):  # Increased clusters for finer granularity
        if np is None or DBSCAN is None:
            raise ImportError("NumPy or scikit-learn not available for NeuromorphicMemory")

        self.memory_dim = memory_dim
        self.num_clusters = num_clusters
        # Use FAISS if available; otherwise fall back to a simple in-memory cosine index
        if faiss is not None:
            self.memory_index = faiss.IndexFlatIP(memory_dim)
            self._uses_faiss = True
        else:
            self._uses_faiss = False
            class _LocalIPIndex:
                def __init__(self, dim: int):
                    self.dim = dim
                    self.vectors = []
                def add(self, vecs):
                    for v in vecs:
                        self.vectors.append(v.astype(np.float32))
                def search(self, query, k):
                    if not self.vectors:
                        scores = np.zeros((1, k), dtype=np.float32)
                        indices = -np.ones((1, k), dtype=np.int64)
                        return scores, indices
                    mat = np.vstack(self.vectors).astype(np.float32)  # (N, D)
                    # Normalize matrix and query for cosine similarity
                    mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
                    q = query.astype(np.float32)
                    q = q / (np.linalg.norm(q) + 1e-8)
                    sims = np.dot(mat_norm, q.T).reshape(-1)  # (N,)
                    top_k = min(k, sims.shape[0])
                    idxs = np.argsort(-sims)[:top_k]
                    scores = sims[idxs].astype(np.float32)
                    # Pad to k
                    if top_k < k:
                        pad_scores = np.zeros(k - top_k, dtype=np.float32)
                        pad_idxs = -np.ones(k - top_k, dtype=np.int64)
                        scores = np.concatenate([scores, pad_scores])
                        idxs = np.concatenate([idxs, pad_idxs])
                    return scores.reshape(1, -1), idxs.reshape(1, -1)
            self.memory_index = _LocalIPIndex(memory_dim)
        self.memory_data = []
        self.memory_metadata = []
        self.cluster_model = DBSCAN(
            eps=0.3, min_samples=1
        )  # Tuned for better clustering
        # Keep a parallel store of embeddings for relevance and cluster heuristics
        self._embedding_store = []

        # Lazy-loaded components with fallback
        self.cognitive_embedder = None
        self.tokenizer = None
        self._is_initialized = False
        self.forgetting_threshold = (
            0.1  # For active forgetting of low-relevance memories
        )

    def _initialize_embedder(self):
        """Initializes the embedding model on first use with fallback."""
        if not self._is_initialized:
            if AutoModel is None or AutoTokenizer is None:
                logger.warning("Transformers not available, using random embeddings")
                self.embed_text = lambda text: np.random.randn(
                    1, self.memory_dim
                ).astype(np.float32)  # Fallback
                self._is_initialized = True
                return

            logger.info("Initializing enhanced Neuromorphic Memory embedder...")
            try:
                self.cognitive_embedder = AutoModel.from_pretrained(
                    "sentence-transformers/all-mpnet-base-v2"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-mpnet-base-v2"
                )
                self._is_initialized = True
                logger.info("Embedder initialized successfully.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize model: {e}. Using random embeddings as fallback."
                )
                self.embed_text = lambda text: np.random.randn(
                    1, self.memory_dim
                ).astype(np.float32)  # Fallback

    def embed_text(self, text):
        """Embed text with normalization."""
        self._initialize_embedder()
        if not self._is_initialized:
            return None

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.cognitive_embedder(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        return embedding / np.linalg.norm(embedding)  # L2 normalization

    def add_memory(self, data, metadata=None):
        """Add memory with relevance scoring and forgetting check."""
        embedding = self.embed_text(str(data))
        if embedding is None:
            return
        # Ensure embedding is float32 and normalized
        embedding = embedding.astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm

        # Compute relevance as similarity to existing memories (max cosine similarity)
        relevance = 1.0
        if self._embedding_store:
            mat = np.vstack(self._embedding_store).astype(np.float32)  # (N, D)
            sims = np.dot(mat, embedding.T).reshape(-1)
            relevance = float(np.max(sims))
        # Active forgetting of very low-relevance items
        if relevance < self.forgetting_threshold:
            logger.debug("Memory forgotten due to low relevance.")
            return

        self.memory_index.add(embedding)
        self.memory_data.append(data)
        self.memory_metadata.append(metadata or {})
        # Track embedding for future relevance/cluster computations
        self._embedding_store.append(embedding.squeeze(0))

        # Periodic clustering with forgetting
        if len(self.memory_data) % 50 == 0:  # More frequent updates
            self.update_clustering()

    def retrieve_similar(self, query, k=10):  # Increased k for broader recall
        """Retrieve similar memories with relevance filtering."""
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return []

        scores, indices = self.memory_index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memory_data):
                rel_score = scores[0][i]  # Inner product as relevance
                if rel_score > 0.5:  # Filter low-relevance
                    results.append(
                        {
                            "data": self.memory_data[idx],
                            "metadata": self.memory_metadata[idx],
                            "relevance": rel_score,
                        }
                    )
        return results

    def update_clustering(self):
        """Update clustering with active forgetting."""
        if len(self.memory_data) > 5:
            embeddings = np.vstack([self.embed_text(str(d)) for d in self.memory_data])
            if embeddings.shape[0] == 0:
                return

            clusters = self.cluster_model.fit_predict(embeddings)
            for i, cluster_id in enumerate(clusters):
                self.memory_metadata[i]["cluster"] = int(cluster_id)

            # Forget outlier clusters
            outlier_mask = clusters == -1
            self.memory_data = [
                d for j, d in enumerate(self.memory_data) if not outlier_mask[j]
            ]
            self.memory_metadata = [
                m for j, m in enumerate(self.memory_metadata) if not outlier_mask[j]
            ]