import os
import chromadb
from typing import List, Any
import numpy as np
import uuid

class VectorStore: 
    def __init__(self, collectionName = "pdf_collection",dir = "../db"):
        self.collectionName = collectionName
        self.dir = dir
        self.client = None
        self.collection = None
        self._initializeStore()

    def _initializeStore(self):
        try:
            os.makedirs(self.dir, exist_ok = True)
            self.client  = chromadb.PersistentClient(path  = self.collectionName)
            self.collection = self.client.get_or_create_collection(name = self.collectionName, metadata={"description":"pdf collection"})


        except Exception as e:
            print("error creating directory: ", e)
            return
        

    def addDocs(self, docs:List[Any], embedding:np.ndarray):
        try:
            ids=[]
            documents =[]
            metadatas =[]
            embeddingList = []

            for i, doc in enumerate(zip(docs, embedding)):
                docId = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(docId)

                metadata = dict(doc.metadata)
                metadatas.append(metadata)

                documents.append(doc.page_content)

                embeddingList.append(embedding.tolist())

                
                self.collection.add(
                    ids = ids,
                    metadatas = metadatas,
                    documents= documents,
                    embeddings=embeddingList,
                )
                
        except Exception as e:
            print("error adding docs to vector store:",e)            