import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize user chat input box
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

class OrderInfoRetriever:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()
        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    def _load_data(self):
        data = pd.read_csv(self.file_path)
        data['Order_ID'] = data['Order_ID'].str.strip().str.lower().str.replace(" ", "")
        return data

    def extract_order_id(self, user_input):
        match = re.search(r'\b(ord\s?\d+)\b', user_input.strip().lower())
        if match:
            return match.group(1).replace(" ", "")
        return None

    def retrieve_order_info(self, order_id):
        if order_id in self.data['Order_ID'].values:
            return self.data[self.data['Order_ID'] == order_id].iloc[0].to_dict()
        return None

    def get_order_details(self, user_input):
        order_id = self.extract_order_id(user_input)
        if order_id:
            order_info = self.retrieve_order_info(order_id)
            if order_info:
                return ', '.join([f"{key.lower().replace('_', ' ')} {value}" for key, value in order_info.items()])
        return None


class FAQSystem:
    def __init__(self, faq_path, api_key):
        self.faq_path = faq_path
        self.api_key = api_key
        self.load_faq_data()
        self.initialize_model()

    def load_faq_data(self):
        self.faq_df = pd.read_csv(self.faq_path, encoding='ISO-8859-1')
        self.questions = self.faq_df['Question'].tolist()
        self.answers = self.faq_df['Answer'].tolist()

    def initialize_model(self):
        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.question_embeddings = self.model.encode(self.questions, normalize_embeddings=True)

    def get_response(self, user_query):
        query_embedding = self.model.encode([user_query], normalize_embeddings=True)
        similarities = util.cos_sim(query_embedding, self.question_embeddings)[0].cpu().numpy()
        top_k_indices = np.argsort(similarities)[::-1]

        for i in top_k_indices:
            if similarities[i] > 0.68:
                return self.answers[i]
        return "I couldn't find a relevant answer. Can you please rephrase your query?"


# Streamlit Interface
st.title("Telegram-Like E-commerce Chatbot")

faq_system = FAQSystem(faq_path="Cleaned_Ecommerce_FAQs.csv", api_key="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")
retriever = OrderInfoRetriever(file_path="CRM.csv")

# Telegram-like styles
st.markdown("""
<style>
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #ccc;
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 10px;
}
.user-message {
    background-color: #0088cc;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: right;
    float: right;
    clear: both;
    max-width: 70%;
}
.bot-message {
    background-color: #eeeeee;
    color: black;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: left;
    float: left;
    clear: both;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)

# Chat container for conversation
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for chat_input in st.session_state.chat_history:
    if chat_input.startswith("You:"):
        st.markdown(f'<div class="user-message">{chat_input[4:]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{chat_input[4:]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input box and send button
col1, col2 = st.columns([4, 1])
with col1:
    chat_input = st.text_area("Type your message here...", key="chat_input", height=50, value=st.session_state.chat_input)
with col2:
    if st.button("Send"):
        if chat_input:
            # Add user query to chat history
            st.session_state.chat_history.append(f"You: {chat_input}")

            # Process user query
            order_info = retriever.get_order_details(chat_input)
            if order_info:
                response = f"Order Info: {order_info}"
            else:
                response = faq_system.get_response(chat_input)

            # Add AI response to chat history
            st.session_state.chat_history.append(f"AI: {response}")

            # Reset chat input for next message
            st.session_state.chat_input = ""  # Clear input box for next message
            st.experimental_rerun()  # Refresh app to show the latest messages
