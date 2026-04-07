from src.embedding import Embedding
from src.vectorStore import VectorStore

class Search:
    def __init__(self, vectorStoreInstance: VectorStore, embeddingInstance: Embedding):
        self.vectorStoreInstance = vectorStoreInstance
        self.embeddingInstance = embeddingInstance

    def search(self, query: str, topk: int = 5, scoreThreshhold: float = 0.5):
        try:
            # chuking
            # embedding
            # vector
            # search in store

            # chunks = self.embeddingInstance.chunking([query])[0]

            embedding = self.embeddingInstance.embedding([query])[0]

            results = self.vectorStoreInstance.collection.query(query_embeddings=[embedding], n_results=topk)

            contextDocs = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                ids = results["ids"][0]
                distance = results["distances"][0]
                metadata = results["metadatas"][0]

                for i, (docId, distance, doc, metadata) in enumerate(zip(ids, distance, documents, metadata)):

                    score = 1-distance

                    if score>= scoreThreshhold:
                        contextDocs.append({
                            "id": docId,
                            "score": score,
                            "doc": doc,
                            "metadata": metadata
                        })

            else: print("No doc found")           

            return contextDocs

        except Exception as e:
            print("error searching vector store: ", e)
