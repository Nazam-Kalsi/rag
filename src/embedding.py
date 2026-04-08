from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_core.documents import Document


class Embedding:
    def __init__(
        self,
        modelName="all-MiniLM-L6-v2",
        chunkSize: int = 1000,
        chunkOverlap: int = 200,
    ):
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.model = SentenceTransformer(modelName)

    def chunking(self, docs: List[Any]) -> List[Any]:
        try:
            textSplitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunkSize,
                chunk_overlap=self.chunkOverlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )

            splittedDocs = textSplitter.split_documents(docs)
            return splittedDocs
            print("splittedDocs: ", splittedDocs)
        except Exception as e:
            print("error chunking: ", e)
            return []

    def embedding(self, splittedDocs:List[str]) -> np.ndarray:
        try:

            embedding = self.model.encode(splittedDocs, show_progress_bar=True)
            return embedding
        except Exception as e:
            print("error embedding: ", e)
            return []


# doc = [
#     Document(
#         metadata={
#             "producer": "Adobe PDF Library 7.0",
#             "creator": "Adobe InDesign CS2 (4.0)",
#             "creationdate": "2008-03-12T00:20:34+11:00",
#             "subject": "Download classic literature as completely free eBooks from Planet eBook.",
#             "author": "Franz Kafka",
#             "moddate": "2008-07-06T18:56:40+10:00",
#             "title": "The Metamorphosis",
#             "trapped": "/False",
#             "source": "D:\\NazamKalsi\\damm\\rag\\data\\pdf.pdf",
#             "total_pages": 77,
#             "page": 76,
#             "page_label": "77",
#         },
#         page_content="\x18\x18Free eBooks at Planet eBoo k.com\nand with especially promising prospects. The greatest im -\nprovement in their situation at this moment, of course, had \nto come from a change of dwelling. Now they wanted to \nrent an apartment smaller and cheaper but better situated \nand generally more practical than the present one, which \nGregor had found. While they amused themselves in this \nway, it struck Mr. and Mrs. Samsa almost at the same mo -\nment how their daughter, who was getting more animated \nall the time, had blossomed recently, in spite of all the trou -\nbles which had made her cheeks pale, into a beautiful and \nvoluptuous young woman. Growing more silent and almost \nunconsciously understanding each other in their glances, \nthey thought that the time was now at hand to seek out a \ngood honest man for her. And it was something of a con -\nfirmation of their new dreams and good intentions when at \nthe end of their journey the daughter first lifted herself up",
#     ),
#     Document(
#         metadata={
#             "producer": "Adobe PDF Library 7.0",
#             "creator": "Adobe InDesign CS2 (4.0)",
#             "creationdate": "2008-03-12T00:20:34+11:00",
#             "subject": "Download classic literature as completely free eBooks from Planet eBook.",
#             "author": "Franz Kafka",
#             "moddate": "2008-07-06T18:56:40+10:00",
#             "title": "The Metamorphosis",
#             "trapped": "/False",
#             "source": "D:\\NazamKalsi\\damm\\rag\\data\\pdf.pdf",
#             "total_pages": 77,
#             "page": 76,
#             "page_label": "77",
#         },
#         page_content="good honest man for her. And it was something of a con -\nfirmation of their new dreams and good intentions when at \nthe end of their journey the daughter first lifted herself up \nand stretched her young body.",
#     ),
# ]

# instance = Embedding()
# em = instance.embedding(doc)
# print(em)
