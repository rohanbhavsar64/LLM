# Streamlit Interface
st.title("E-commerce Customer Support Chatbot")

faq_system = FAQSystem(faq_path="Cleaned_Ecommerce_FAQs.csv", api_key="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")
retriever = OrderInfoRetriever(file_path="CRM.csv")

# Function to process user queries and update the chat
def process_chat(user_input):
    global order_info
    if retriever.get_intent(user_input):
        order_info = retriever.get_order_details(user_input)
        if order_info:
            context, refined_answer = faq_system.process_user_query(user_input, order_info)
            return refined_answer
    else:
        context, refined_answer = faq_system.process_user_query(user_input, order_info)
        return refined_answer

# Create a new input tab dynamically for each query
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "queries" not in st.session_state:
    st.session_state.queries = []

# Loop to handle multiple input tabs
for i in range(st.session_state.query_count + 1):
    user_input = st.text_input(f"Query {i + 1}:", key=f"input_{i}", placeholder="Ask your question here...")

    if user_input:
        # Add user input to the query list
        st.session_state.queries.append(user_input)
        st.session_state.chat_history.append(f"You: {user_input}")

        # Process the query
        response = process_chat(user_input)
        st.session_state.chat_history.append(f"AI: {response}")
        st.success(response)

        # Increase the query count to add a new input tab
        st.session_state.query_count += 1

# Display the chat history
st.write("### Conversation History")
for entry in st.session_state.chat_history:
    st.write(entry)
