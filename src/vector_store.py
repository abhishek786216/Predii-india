import os

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_COLLECTION = "vehicle_specs"


class VectorStore:

    def __init__(self, collection_name=CHROMA_COLLECTION):
        missing_variables = [
            name
            for name, value in (
                ("CHROMA_API_KEY", CHROMA_API_KEY),
                ("CHROMA_TENANT", CHROMA_TENANT),
                ("CHROMA_DATABASE", CHROMA_DATABASE),
            )
            if not value
        ]
        if missing_variables:
            raise RuntimeError(
                "Set the following environment variables before creating a "
                f"VectorStore: {', '.join(missing_variables)}."
            )

        self.client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

    def add_chunks(self, chunks):
        if not chunks:
            return

        start_id = self.collection.count()
        batch_size = 250

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]
            ids = [
                f"chunk_{start_id + batch_start + i}"
                for i in range(len(batch))
            ]
            documents = [chunk["text"] for chunk in batch]
            metadatas = [
                {
                    "page": chunk["page"],
                    "method": chunk["method"]
                }
                for chunk in batch
            ]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

    def search(self, query, top_k=5):
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(
                top_k,
                self.collection.count()
            )
        )

        output = []

        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return output