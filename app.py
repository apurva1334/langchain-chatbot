import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ─── Load environment variables ───────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangChain ChatBot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangChain ChatBot")
st.caption("Powered by OpenAI GPT-3 & LangChain")

# ─── Initialize session state ─────────────────────────────────────────────────
# Streamlit reruns the script on every interaction, so we store state
if "conversation" not in st.session_state:
    # LLM setup: text-davinci-003 model, temperature controls creativity
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="text-davinci-003",
        temperature=0.7
    )

    # Memory: keeps track of the full conversation history
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=False
    )

    # ConversationChain: ties LLM + memory together
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of {"role": ..., "content": ...}

# ─── Display chat history ─────────────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Chat input ───────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Get AI response
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.conversation.predict(input=user_input)
        except Exception as e:
            response = f"❌ Error: {str(e)}"

    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save AI response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })

# ─── Sidebar: Clear chat button ───────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        # Reset memory too
        st.session_state.conversation.memory.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Model:** `text-davinci-003`")
    st.markdown("**Memory:** ConversationBufferMemory")
    st.markdown("**Framework:** LangChain + Streamlit")