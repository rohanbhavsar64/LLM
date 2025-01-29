import streamlit as st
from streamlit_chat import message
import tempfile
import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve Hugging Face API token from .env
HUGGINGFACEHUB_API_TOKEN = "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"

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("Hugging Face API token is missing. Please check your .env file.")

DB_FAISS_PATH = 'vectorstore/db_faiss'

def load_llm():
    repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        repo_id=repo_id,
        temperature=0.7,
        max_new_tokens=500
    )
    return llm

st.title("Chat with CSV using Falcon-7B 🦙🦜")
st.markdown("<h3 style='text-align: center; color: white;'>Built by D4X ❤️</h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = CSVLoader(file_path=tmp_file_path, encoding="ISO-8859-1", csv_args={'delimiter': ','})
        data = loader.load()
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        data = []

    if data:
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        
        if not os.path.exists(DB_FAISS_PATH):
            os.makedirs(DB_FAISS_PATH)

        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)

        llm = load_llm()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        def conversational_chat(query):
            # Retrieve relevant context from the CSV
            try:
                # Retrieve the conversation history in the correct tuple format
                context_docs = chain.invoke({"question": query, "chat_history": st.session_state['history']})
                context_text = "\n".join([doc.page_content for doc in context_docs.get('source_documents', [])])
            except Exception as e:
                context_text = ""
                st.error(f"Error in retrieval: {e}")

            # Debugging the retrieved context
            st.write("Context Retrieved:", context_text)

            # Craft the prompt for the LLM, including both the conversation history and context
            history_text = "\n".join([f"User: {item[0]}\nAgent: {item[1]}" for item in st.session_state["history"]])

            if context_text and history_text:
                prompt_template = (
                    f"You are a helpful and empathetic assistant. Respond to the user in a natural, human-like tone, "
                    f"using both the conversation history and the relevant CSV data to craft your response.\n\n"
                    f"Conversation History:\n{history_text}\n\n"
                    f"Relevant CSV Data:\n{context_text}\n\n"
                    f"User Query: {query}\n\n"
                    f"Answer naturally, as if you're explaining to someone in Hinglish (a mix of Hindi and English):"
                )
            elif context_text:
                prompt_template = (
                    f"You are a helpful and empathetic assistant. Respond to the user in a natural, human-like tone, "
                    f"using the relevant CSV data to craft your response.\n\n"
                    f"Relevant CSV Data:\n{context_text}\n\n"
                    f"User Query: {query}\n\n"
                    f"Answer naturally, as if you're explaining to someone in Hinglish (a mix of Hindi and English):"
                )
            elif history_text:
                prompt_template = (
                    f"You are a helpful and empathetic assistant. Respond to the user in a natural, human-like tone, "
                    f"using the conversation history to craft your response.\n\n"
                    f"Conversation History:\n{history_text}\n\n"
                    f"User Query: {query}\n\n"
                    f"Answer naturally, as if you're explaining to someone from your side."
                )
            else:
                prompt_template = (
                    f"You are a helpful and empathetic assistant. Respond to the user in a natural, human-like tone.\n\n"
                    f"User Query: {query}\n\n"
                    f"Answer naturally, as if you're explaining to someone in Hinglish (a mix of Hindi and English):"
                )

            # Generate response using the LLM
            response = llm(prompt_template)

            # Store the current query and response in conversation history (as a tuple)
            st.session_state["history"].append((query, response))
            return response

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]
        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
