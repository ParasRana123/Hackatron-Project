import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
import networkx as nx
import re

# Define the dream interpretation prompt
prompt_template = PromptTemplate(
    input_variables=["dream_description"],
    template="""
    You are a dream interpreter. Analyze the following dream based on psychological theories and cultural references:
    
    Dream: {dream_description}
    
    Provide a detailed interpretation.
    """
)

api_key = "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"
# Initialize the Groq model
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=api_key)
dream_interpreter_chain = LLMChain(llm=llm, prompt=prompt_template)

def interpret_dream(dream_description):
    return dream_interpreter_chain.run(dream_description)

# Function to create a theme mapping graph
def create_theme_mapping(text):
    words = re.findall(r'\w+', text.lower())
    themes = set(words)  # Unique themes based on words

    # Create a graph
    G = nx.Graph()
    for word in themes:
        G.add_node(word)  # Add nodes for each theme
        for related_word in themes:
            if word != related_word:  # Avoid self-loops
                G.add_edge(word, related_word)  # Connect every word to every other word

    return G

# Streamlit app layout
st.title("Dream Interpretation Chatbot")
st.write("Share your dream below, and I'll provide an interpretation based on psychological theories and cultural references.")

# User input
user_dream = st.text_area("Describe your dream:", height=150)
dream_category = st.selectbox("Categorize your dream:", ["Select Category", "Nightmare", "Lucid Dream", "Recurring Dream", "Anxiety Dream", "Adventure", "Other"])

if st.button("Interpret Dream"):
    if user_dream and dream_category != "Select Category":
        interpretation = interpret_dream(user_dream)
        st.subheader("Interpretation:")
        st.write(interpretation)

        # Create a theme mapping graph for visualization
        st.subheader("Theme Mapping:")
        theme_graph = create_theme_mapping(user_dream)
        
        # Draw the graph using networkx
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(theme_graph)
        nx.draw(theme_graph, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold', edge_color='gray')
        plt.title('Theme Mapping of Your Dream')
        plt.tight_layout()
        st.pyplot(plt)
        
        st.success(f"Your dream has been categorized as: **{dream_category}**")
    else:
        st.warning("Please enter a dream description and select a category to interpret.")