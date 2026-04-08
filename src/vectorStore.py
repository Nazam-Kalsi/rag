import os
import chromadb
from typing import List, Any
import numpy as np
import uuid
from langchain_core.documents import Document

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

            # print("doc: ", docs)
            for i, (doc, emb) in enumerate(zip(docs, embedding)):
                docId = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(docId)

                metadata = dict(doc.metadata)
                metadatas.append(metadata)

                documents.append(doc.page_content)

                embeddingList.append(emb.tolist())

                
                self.collection.add(
                    ids = ids,
                    metadatas = metadatas,
                    documents= documents,
                    embeddings=embeddingList,
                )
                
        except Exception as e:
            print("error adding docs to vector store:",e)            

# doc = [Document(metadata={'producer': 'Adobe PDF Library 7.0', 'creator': 'Adobe InDesign CS2 (4.0)', 'creationdate': '2008-03-12T00:20:34+11:00', 'subject': 'Download classic literature as completely free eBooks from Planet eBook.', 'author': 'Franz Kafka', 'moddate': '2008-07-06T18:56:40+10:00', 'title': 'The Metamorphosis', 'trapped': '/False', 'source': 'D:\\NazamKalsi\\damm\\rag\\data\\pdf.pdf', 'total_pages': 77, 'page': 75, 'page_label': '76'}, page_content='the window and remained there, with their arms about each \nother. Mr. Samsa turned around in his chair in their direc -\ntion and observed them quietly for a while. Then he called \nout, ‘All right, come here then. Let’s finally get rid of old \nthings. And have a little consideration for me.’ The women \nattended to him at once. They rushed to him, caressed him, \nand quickly ended their letters.\nThen all three left the apartment together, something \nthey had not done for months now, and took the electric \ntram into the open air outside the city. The car in which they \nwere sitting by themselves was totally engulfed by the warm \nsun. They talked to each other, leaning back comfortably \nin their seats, about future prospects, and they discovered \nthat on closer observation these were not at all bad, for all \nthree had employment, about which they had not really \nquestioned each other at all, which was extremely favorable'), Document(metadata={'producer': 'Adobe PDF Library 7.0', 'creator': 'Adobe InDesign CS2 (4.0)', 'creationdate': '2008-03-12T00:20:34+11:00', 'subject': 'Download classic literature as completely free eBooks from Planet eBook.', 'author': 'Franz Kafka', 'moddate': '2008-07-06T18:56:40+10:00', 'title': 'The Metamorphosis', 'trapped': '/False', 'source': 'D:\\NazamKalsi\\damm\\rag\\data\\pdf.pdf', 'total_pages': 77, 'page': 76, 'page_label': '77'}, page_content='\x18\x18Free eBooks at Planet eBoo k.com\nand with especially promising prospects. The greatest im -\nprovement in their situation at this moment, of course, had \nto come from a change of dwelling. Now they wanted to \nrent an apartment smaller and cheaper but better situated \nand generally more practical than the present one, which \nGregor had found. While they amused themselves in this \nway, it struck Mr. and Mrs. Samsa almost at the same mo -\nment how their daughter, who was getting more animated \nall the time, had blossomed recently, in spite of all the trou -\nbles which had made her cheeks pale, into a beautiful and \nvoluptuous young woman. Growing more silent and almost \nunconsciously understanding each other in their glances, \nthey thought that the time was now at hand to seek out a \ngood honest man for her. And it was something of a con -\nfirmation of their new dreams and good intentions when at \nthe end of their journey the daughter first lifted herself up'), Document(metadata={'producer': 'Adobe PDF Library 7.0', 'creator': 'Adobe InDesign CS2 (4.0)', 'creationdate': '2008-03-12T00:20:34+11:00', 'subject': 'Download classic literature as completely free eBooks from Planet eBook.', 'author': 'Franz Kafka', 'moddate': '2008-07-06T18:56:40+10:00', 'title': 'The Metamorphosis', 'trapped': '/False', 'source': 'D:\\NazamKalsi\\damm\\rag\\data\\pdf.pdf', 'total_pages': 77, 'page': 76, 'page_label': '77'}, page_content='good honest man for her. And it was something of a con -\nfirmation of their new dreams and good intentions when at \nthe end of their journey the daughter first lifted herself up \nand stretched her young body.')]

# instance = VectorStore()
# instance.addDocs(doc, np.random.rand(3,384))