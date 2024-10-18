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

import validators.url

# Page configuration
st.set_page_config(page_title="Triumo Chatbot: App With Multiple Functionalities")

api_key = "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"

# Option Selection
st.title("Triumo Chatbot: App With Multiple Functionalities")
st.subheader("Choose an option to proceed:")