from src.dataLoader import loadDocs
from src.embedding import Embedding
from src.llmResult import generateAns, llmModel, search
from src.vectorStore import VectorStore
from src.search import Search

def app(query:str):
    docs = loadDocs("data")
    print("docs: ", docs)

    embeddingInstance = Embedding()
    vectorStoreInstance = VectorStore()

    chunks = embeddingInstance.chunking(docs)
    print("chunks: ", chunks)
    embeddings = embeddingInstance.embedding([doc.page_content for doc in chunks])

    vectorStoreInstance.addDocs(chunks, embeddings)

    searchInstance = Search(vectorStoreInstance, embeddingInstance)    

    ans = searchInstance.search(query, topk=5, scoreThreshhold=0.5)
    return ans


if __name__ == "__main__":
    
    ans = app("What is the main topic of the document?")
    print("ans: ",ans)



        
