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

# Global variables
order_info = None


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
        global order_info
        if order_id in self.data['Order_ID'].values:
            order_info = self.data[self.data['Order_ID'] == order_id].iloc[0].to_dict()
            return order_info
        return None

    def format_order_info(self):
        global order_info
        if order_info:
            return ', '.join([f"{key.lower().replace('_', ' ')} {value}" for key, value in order_info.items()])
        return "No order information available."

    def get_order_details(self, user_input):
        order_id = self.extract_order_id(user_input)
        if order_id:
            if self.retrieve_order_info(order_id):
                formatted_info = self.format_order_info()
                return formatted_info
        return None

    def get_intent(self, query):
        order_id = self.extract_order_id(query)
        return order_id is not None


class FAQSystem:
    def __init__(self, faq_path, api_key):
        self.faq_path = faq_path
        self.api_key = api_key
        self.load_faq_data()
        self.initialize_model()

    def load_faq_data(self):
        try:
            self.faq_df = pd.read_csv(self.faq_path, encoding='ISO-8859-1')
            self.questions = self.faq_df['Question'].tolist()
            self.answers = self.faq_df['Answer'].tolist()
        except Exception as e:
            st.error(f"Error loading FAQ data: {e}")
            self.faq_df = pd.DataFrame(columns=['Question', 'Answer'])
            self.questions = []
            self.answers = []

    def initialize_model(self):
        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.question_embeddings = self.model.encode(self.questions, normalize_embeddings=True)

    def get_response(self, user_query):
        query_embedding = self.model.encode([user_query], normalize_embeddings=True)
        similarities = util.cos_sim(query_embedding, self.question_embeddings)[0].cpu().numpy()
        top_k_indices = np.argsort(similarities)[::-1]

        context_answers = []
        for i in top_k_indices:
            similarity_score = similarities[i]
            if similarity_score > 0.68:
                context_answers.append(self.answers[i])
            else:
                break

        context = " ".join(context_answers) if context_answers else "No relevant answers found."
        return context

    def process_user_query(self, user_query, order_info=None):
        context = self.get_response(user_query)

        if order_info:
            context = f"Order Information: {order_info}. " + context

        india_tz = pytz.timezone('Asia/Kolkata')
        india_datetime = datetime.now(india_tz).strftime('%Y-%m-%d %H:%M:%S')

        refined_answer = self.refine_answer_with_falcon(user_query, context, india_datetime)
        return context, refined_answer

    def refine_answer_with_falcon(self, user_query, context, india_datetime):
        # Combine chat history into a single formatted string
        chat_history_text = "\n".join(st.session_state.chat_history)

        if context == "No relevant answers found.":
            prompt = (
                "You are a helpful customer care agent. You must remember and use the entire previous conversation "
                "to provide coherent and context-aware responses. When users ask casual or conversational questions, "
                "respond with polite and conversational replies.\n"
                f"Conversation History:\n{chat_history_text}\n"
                f"User Query: {user_query}\n"
                f"Indian Date and Time: {india_datetime}\n"
                "Answer (Based on the entire conversation history, provide a concise and polite response):"
            )
        else:
            prompt = (
                "You are a well-experienced e-commerce customer service representative. You must remember and use the "
                "entire previous conversation to provide contextually appropriate responses. "
                "Focus on addressing the user's query directly while maintaining professionalism.\n"
                f"Conversation History:\n{chat_history_text}\n"
                f"User Query: {user_query}\n"
                f"Context: {context}\n"
                f"Indian Date and Time: {india_datetime}\n"
                "Answer (Based on the provided context and entire conversation history, give a concise and helpful response):"
            )

        api_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 300,
                "temperature": 0.6,
                "top_p": 0.9,
                "no_repeat_ngram_size": 2,
            }
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            refined_answer = response.json()[0]["generated_text"]
        except Exception as e:
            refined_answer = f"Error refining answer: {e}"

        refined_answer = refined_answer.replace(prompt, "").strip()
        return refined_answer


# Streamlit Interface
st.title("E-commerce Customer Support Chatbot")

faq_system = FAQSystem(faq_path="Cleaned_Ecommerce_FAQs.csv", api_key="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")
retriever = OrderInfoRetriever(file_path="CRM.csv")

user_input = st.text_input("You:", placeholder="Ask your question here...")

if user_input:
    # Add user input to chat history
    st.session_state.chat_history.append(f"You: {user_input}")

    if retriever.get_intent(user_input):
        order_info = retriever.get_order_details(user_input)
        if order_info:
            context, refined_answer = faq_system.process_user_query(user_input, order_info)
            # Add AI response to chat history
            st.session_state.chat_history.append(f"AI: {refined_answer}")
            st.success(refined_answer)
    else:
        context, refined_answer = faq_system.process_user_query(user_input, order_info)
        st.session_state.chat_history.append(f"AI: {refined_answer}")
        st.success(refined_answer)

# Display the chat history
st.write("### Conversation History")
for entry in st.session_state.chat_history:
    st.write(entry)
