class Retriever:

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5):
        results = self.vector_store.search(
            query,
            top_k=top_k
        )

        return results