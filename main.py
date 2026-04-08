from src.dataLoader import loadDocs
from src.embedding import Embedding
from src.llmResult import generateAns
from src.vectorStore import VectorStore
from src.search import Search
from src.llmResult import generateAns
def app(query:str):
    # docs = loadDocs("data")
    # # print("docs: ", docs)

    embeddingInstance = Embedding()
    vectorStoreInstance = VectorStore()
    searchInstance = Search(vectorStoreInstance, embeddingInstance)    

    # chunks = embeddingInstance.chunking(docs) # convert pdf content into documents
    # # print("chunks: ", chunks)
    # text = [ doc.page_content for doc in chunks]
    # embeddings = embeddingInstance.embedding(text)
    # # print("embeddings: ",embeddings)

    # vectorStoreInstance.addDocs(chunks, embeddings)


    # ans = searchInstance.search(query, topk=5, scoreThreshhold=0.5)
    ans = generateAns(query, searchInstance)
    return  ans


if __name__ == "__main__":
    
    ans = app("I cannot make you understand. I cannot make anyone understand what is happening inside me. I cannot even explain it to myself")

 

    # ans = generateAns("How about if I sleep a little bit longer and forget all this nonsense")


    print("ans: ",ans)



        
