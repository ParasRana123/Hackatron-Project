import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader, UnstructuredURLLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.agents import initialize_agent, AgentType
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain.chains import create_history_aware_retriever , create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import io
from fpdf import FPDF
import requests
import json
import validators
import time
import re
from gtts import gTTS

import validators.url

# Page configuration
st.set_page_config(page_title="Triumo Chatbot: App With Multiple Functionalities")

api_key = "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"

# Option Selection
st.title("Triumo Chatbot: App With Multiple Functionalities")
st.subheader("Choose an option to proceed:")

# Options
option = st.selectbox("Choose an action:", ["Select", "Ask Queries" , "Summarise the PDF And Ask Questions"])

# Initialize the Groq model
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=api_key)

# Initialise the type of embeddings
embeddings = OllamaEmbeddings()
retriever_tool = None

# Initializing the Wikipedia and Arxiv tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchResults(name="Search")

retriever_tool = None
embeddings = OllamaEmbeddings()

# ------------------------- Normal Chatbot Functonality ------------------------------ #
if option == "Chat With LLM":
    # Initialize session state for messages (conversation history)
  if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search the web. How can I assist you?"}
    ]

  # Display previous messages in the chat
  for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

  # Handling the user prompt
  if prompt := st.chat_input(placeholder="Ask anything..."):
    # Add the user's message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize the model and tools
    llm = ChatGroq(api_key=api_key, model="gemma2-9b-it", streaming=True)

    # Include available tools in the agent
    tools = [arxiv, wiki, search]

    # Create the search agent
    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION , max_iterations=15   # Increase this value as per your needs
    )

    # Construct the conversation history as a single string
    conversation_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
    )

    # Run the agent with the full conversation history
    response = search_agent.run(conversation_history)

    # Add assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

# ------------------- Summarize a PDF with audio-based summarisation and Ask Queries -------------------#
# Summarize PDF option
elif option == "Summarise the PDF And Ask Questions":
    session_id = st.text_input("Session Id:", value="default Session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    # taking the PDF from the user to summarise it
    uploaded_files = st.file_uploader("Choose a PDF file:", type="pdf", accept_multiple_files=True)
    documents = []
    
    if uploaded_files:
        with st.spinner("Processing the PDF files, Please wait..."):
            for uploaded_file in uploaded_files:
                temppdf = f"./{uploaded_file.name}"  # Unique name for each file
                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.getvalue())
                
                loader = PyPDFLoader(temppdf)
                docs = loader.load_and_split()
                documents.extend(docs)  # Use extend to flatten the list of docs

        # Splitting the documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        final_documents = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=final_documents, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Creating a prompt template for summarization
        chunks_prompt = """
        Write a short and concise summary of the following speech,
        Speech: {text}
        Summary:
        """
        map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

        final_prompt = """
        Provide the final summary of the entire speech with these important points.
        Add a motivational Title, Start the precise summary with an introduction and provide the summary in numbered points for the speech,
        Speech: {text}
        """
        final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)

        # Load summarization chain
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=final_prompt_template,
            verbose=True
        )

        # Run the summarization chain
        output = summary_chain.run(final_documents)
        st.success(output)
        tts = gTTS(output , lang='en')
        audio_file = "summary.mp3"
        tts.save(audio_file)

        # adding an audio player
        st.audio(audio_file , format='audio/mp3')

        # Define the retriever for the RAG chain
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. Do not answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer Question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer say that you don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Function to get the session history from session state
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Taking user input for question-answering
        user_input = st.text_input("Your Question:")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )

            # Display conversation history and answer
            # st.write(f"Session ID: {session_id}")
            st.write(response['answer'])
            # st.write("Chat History:", session_history.messages)
