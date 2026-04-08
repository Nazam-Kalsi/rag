from src.embedding import Embedding
from src.vectorStore import VectorStore
# if i use this class in main.py then i have tom import as src.<class> and if i have to directly execute this file to test the functionality then i have to import as <class> without src. coz this file is in src folder.
class Search:
    def __init__(self, vectorStoreInstance: VectorStore, embeddingInstance: Embedding):
        self.vectorStoreInstance = vectorStoreInstance
        self.embeddingInstance = embeddingInstance

    def search(self, query: str, topk: int = 5, scoreThreshhold: float = 0.5):
        try:
            # chunking
            # embedding
            # vector
            # search in store

            # chunks = self.embeddingInstance.chunking([query])[0]

            embedding = self.embeddingInstance.embedding([query])[0]
            # print("embedding: ", embedding)

            results = self.vectorStoreInstance.collection.query(query_embeddings=[embedding], n_results=topk)
            # print("results: ", results)
            contextDocs = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                ids = results["ids"][0]
                distance = results["distances"][0]
                metadata = results["metadatas"][0]

                for i, (docId, distance, doc, metadata) in enumerate(zip(ids, distance, documents, metadata)):
                    # distance:  [[1.2601248025894165, 1.2601248025894165, 1.2601248025894165, 1.2601248025894165, 1.2601248025894165]]
                    score = 1-distance
                    print(score)
                    #score:  -0.2601248025894165
                    # if score>= scoreThreshhold:
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

# v = VectorStore()
# e = Embedding()
# s = Search(v,e)
# a = s.search("How about if I sleep a little bit longer and forget all this nonsense")
# print(a)

