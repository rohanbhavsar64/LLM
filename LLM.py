import os
import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Set HuggingFace API token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# Initialize the LLM
h = HuggingFaceHub(repo_id='google/flan-t5-large', model_kwargs={'temperature': 0.9})

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# File path for FAISS index
file_path = 'faiss_index'

# Function to create and save the FAISS database
def create_db():
    loader = CSVLoader(file_path='Cleaned_Ecommerce_FAQs.csv', source_column='Question')
    data = loader.load()
    db = FAISS.from_documents(documents=data, embedding=embeddings)
    db.save_local(file_path)

# Load the FAISS vector database
vector_db = FAISS.load_local(file_path, embeddings)

# Create retriever
r = vector_db.as_retriever()

# Create the QA chain
chain = RetrievalQA.from_chain_type(llm=h, chain_type='stuff', retriever=r, input_key='query', return_source_documents=True)

# Streamlit input and output
q = st.text_input("Ask a question:")
if q:
    result = chain({"query": q})
    st.write(result['result'])
