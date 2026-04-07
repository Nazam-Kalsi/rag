from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedding:
    def __init__(self, modelName="all-MiniLM-L6-v2",chunkSize:int=1000, chunkOverlap:int =200):
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.model = SentenceTransformer(modelName)

    def chunking(self, docs:List[Any])-> List[Any]:
        try:
            textSplitter = RecursiveCharacterTextSplitter(
                chunk_size = self.chunkSize,
                chunk_overlap = self.chunkOverlap,
                length_function = len,
                separators=["\n\n", "\n", " ", ""]
            )

            splittedDocs = textSplitter.split_documents(docs)
            return splittedDocs
            print("splittedDocs: ",splittedDocs)
        except Exception as e:
            print("error chunking: ", e)
            return []
    
    def embedding(self, splittedDocs:List[str])->np.ndarray:
        try:
            
            text = [doc.page_content  for doc in splittedDocs]
            embedding = self.model.encode(text,show_progress_bar=True)
            return embedding
        except Exception as e:
            print("error embedding: ", e)
            return []