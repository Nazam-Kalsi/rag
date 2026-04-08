from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from src.search import Search
from src.embedding import Embedding
from src.vectorStore import VectorStore
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

llmModel= GoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# embeddingInstance = Embedding()
# vectorStoreInstance = VectorStore()
# search = Search(embeddingInstance, vectorStoreInstance)

def generateAns(query, search, topK = 5):
    try:
        result = search.search(query,topK)
        # print("result: ",result)
        content = "\n\n".join([doc["doc"]for doc in result]) if result else "no revelant context found"
        prompt = f"Answer the question based on the following retrieved content:\n\n{content}\n\nQuestion: {query}\nAnswer:"

        answer = llmModel.invoke([prompt.format(content=content, query=query)])
        return answer
    except Exception as e:
        print("Error while generation error: ",e)    
