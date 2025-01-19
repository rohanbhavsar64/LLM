import os
import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import sentence_transformers
os.environ['HUGGINGFACEHUB_API_TOKEN']='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'
h=HuggingFaceHub(repo_id='google/flan-t5-large',model_kwargs={'temperature':0.9})
embeddings=HuggingFaceEmbeddings()
file_path='faiss_index'
def create_db():
  loader=CSVLoader(file_path='Cleaned_Ecommerce_FAQs.csv',source_column='Question')
  data=loader.load()
  db=FAISS.from_documents(documents=data,embedding=embeddings)
  db.save_local(file_path)
vector_db=FAISS.load_local(file_path, embeddings)
r=vector_db.as_retriever()
chain=RetrievalQA.from_chain_type(llm=h,chain_type='stuff',retriever=r,input_key='query',return_source_documents=True)
q=st.text_input()
if q:
  st.write(chain(q)['result'])
