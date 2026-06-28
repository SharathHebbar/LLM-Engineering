import os
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

import faiss
import redis


class CacheManager:

    def __init__(self, redis_client, model=None, faiss_index_path='faiss.index', DIM=384):
        self.redis_client = redis_client
        self.faiss_index_path = faiss_index_path
        self.DIM = DIM
        self.model = model

        # Load or initialize FAISS index
        if os.path.exists(self.faiss_index_path):
            self.index = faiss.read_index(self.faiss_index_path)
        else:
            self.index = faiss.IndexFlatL2(self.DIM)  # Assuming 384-dimensional embeddings

        # Mapping from FAISS ID to Redis key
        self.faiss_id_to_redis = {}

    def _normalize(self, vector):
        return vector / np.linalg.norm(vector)
    
    def embed(self, text):
        vec = self.model.encode([text])
        self.normalized_vector = self._normalize(vec).astype('float32')
    
    def hash_question(self, question: str):
        self.redis_key = f"cache:query:{hashlib.sha256(question.lower().encode()).hexdigest()}"

    
    def get_exact_cache(self, question: str):
        self.hash_question(question)
        return self.redis_client.get(self.redis_key)


    def store_cache(self, question: str, answer: str):
        self.hash_question(question)
        self.redis_client.set(self.redis_key, answer, ex=300)

        self.embed(question)
        
        idx = self.index.ntotal
        self.index.add(self.normalized_vector)

        self.faiss_id_to_redis [idx] = self.redis_key

    
    def get_semantic_cache(self, question: str, threshold=0.85):
        if self.index.ntotal == 0:
            return None
        
        self.embed(question)
        
        D, I = self.index.search(self.normalized_vector, k=1)

        score = float(D[0][0])
        idx = int(I[0][0])

        if score >= threshold:
            redis_key = self.faiss_id_to_redis[idx]
            if redis_key:
                return self.redis_client.get(self.redis_key)
        return None