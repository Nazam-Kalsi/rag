from pathlib import Path
from typing import List, Dict, Any
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, JSONLoader
from langchain_community.document_loaders.excel import  UnstructuredExcelLoader

def loadDocs(path:str)->List[Any]:
    try:
        dataPath = Path(path).resolve() 
        print("data path: ", dataPath)# read path in logs for corrent path for the darta folder

        docs = []

        files = list(dataPath.glob("**/*.pdf")) # glob is used to find all the files with the given extension in the directory and its subdirectories
        print("files: ", len(files))

        for file in files:
            loader = PyPDFLoader(str(file))
            doc = loader.load()
            docs.extend(doc)
        
        return docs
    except Exception as e:
        print("error resolving path: ", e)
        return []
    
    
