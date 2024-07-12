#import Essential dependencies

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

#create a new file named vectorstore in your current directory.
if __name__=="__main__":
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        loader=PyPDFLoader("./random machine learing pdf.pdf")
        docs=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api_key))
        vectorstore.save_local(DB_FAISS_PATH)


        print(f"Initializing FAQ vectorDB")
        FAQ_DB_FAISS_PATH = 'vectorstore/faq_db_faiss'
        sample_faq = pd.read_excel('faq_timestamp.xlsx')
        texts_faq = sample_faq['Original Question'].to_list()
        metadatas = [{'index':i} for i in range(len(texts_faq))]
        db_faq = FAISS.from_texts(texts_faq, OpenAIEmbeddings(api_key=openai_api_key), metadatas = metadatas)
        db_faq.save_local(FAQ_DB_FAISS_PATH)
        print(f"Done initializing FAQ vectorDB")