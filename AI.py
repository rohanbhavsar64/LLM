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

# Initialize user query input box
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

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
st.title("WhatsApp-Like E-commerce Chatbot")

faq_system = FAQSystem(faq_path="Cleaned_Ecommerce_FAQs.csv", api_key="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")
retriever = OrderInfoRetriever(file_path="CRM.csv")

# Chat Interface
st.markdown("""
<style>
    .user-message {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    .bot-message {
        background-color: #ECECEC;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Display the conversation
for message in st.session_state.chat_history:
    if message.startswith("You:"):
        st.markdown(f'<div class="user-message">{message[4:]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{message[4:]}</div>', unsafe_allow_html=True)

# Textbox for input message
user_input = st.text_area("Type your message here...", key="input_box", height=100)

# Handle sending the message
if st.button("Send"):
    if user_input:
        # Add user query to chat history
        st.session_state.chat_history.append(f"You: {user_input}")

        # Process user query
        order_info = retriever.get_order_details(user_input)
        if order_info:
            response = f"Order Info: {order_info}"
        else:
            response = faq_system.get_response(user_input)

        # Add AI response to chat history
        st.session_state.chat_history.append(f"AI: {response}")

        # Clear the input box after submission
        st.session_state.user_input = ""  # Reset the text area value
        st.rerun()  # Rerun the app to update the chat

# Display the chat history again to reflect the latest messages
st.write("### Conversation History")
for message in st.session_state.chat_history:
    if message.startswith("You:"):
        st.markdown(f'<div class="user-message">{message[4:]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{message[4:]}</div>', unsafe_allow_html=True)
