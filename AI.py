import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
from datetime import datetime
import pytz

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

    def refine_answer(self, user_query, context=None):
        chat_history_text = "\n".join(st.session_state.chat_history)
        prompt = (
            f"Conversation History:\n{chat_history_text}\n"
            f"User Query: {user_query}\n"
            "Answer:"
        )

        api_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": prompt, "parameters": {"max_length": 300}}

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()[0]["generated_text"].strip()
        except Exception:
            return "Sorry, I'm unable to process your query at the moment."


# Streamlit Interface
st.title("Telegram-Like E-commerce Chatbot")

faq_system = FAQSystem(faq_path="Cleaned_Ecommerce_FAQs.csv", api_key="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")
retriever = OrderInfoRetriever(file_path="CRM.csv")

# Styling for Telegram-Like Chat
st.markdown("""
<style>
    .chat-container {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        max-height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #dcf8c6;
        padding: 8px 12px;
        border-radius: 15px;
        margin-bottom: 8px;
        text-align: right;
        width: fit-content;
        margin-left: auto;
    }
    .bot-message {
        background-color: #eaeaea;
        padding: 8px 12px;
        border-radius: 15px;
        margin-bottom: 8px;
        text-align: left;
        width: fit-content;
        margin-right: auto;
    }
    .input-container {
        display: flex;
        margin-top: 10px;
    }
    .input-box {
        flex: 1;
        margin-right: 10px;
        padding: 10px;
        border-radius: 15px;
        border: 1px solid #ddd;
    }
    .send-button {
        background-color: #0088cc;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 15px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Display the chat conversation
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for chat_input in st.session_state.chat_history:
    if chat_input.startswith("You:"):
        st.markdown(f'<div class="user-message">{chat_input[4:]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{chat_input[4:]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input box and send button
st.markdown('<div class="input-container">', unsafe_allow_html=True)
chat_input = st.text_area("", key="chat_input", height=40, label_visibility="collapsed")
send = st.button("Send", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

# Process user query if "Send" is clicked
if send and chat_input:
    # Add user query to chat history
    st.session_state.chat_history.append(f"You
