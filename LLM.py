import os
import streamlit
from langchain_community.llms import HuggingFaceHub
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
os.environ['HUGGINGFACEHUB_API_TOKEN']='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'
h=HuggingFaceHub(repo_id='google/flan-t5-large',model_kwargs={'temperature':0.9})
loader=CSVLoader(file_path='Cleaned_Ecommerce_FAQs.csv',source_column='Question')
data=loader.load()
embeddings=HuggingFaceEmbeddings()
db=FAISS.from_documents(documents=data,embedding=embeddings)
r=db.as_retriever()
chain=RetrievalQA.from_chain_type(llm=h,chain_type='stuff',retriever=r,input_key='query',return_source_documents=True)
st.write(chain('If I cancel my order for that do you have mid cancelation refoud policy?')['result'])
