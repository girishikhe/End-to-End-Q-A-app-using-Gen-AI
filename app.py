import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

# LangSmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")  

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user query."),
    ("user", "Question: {question}")
])

def generate_response(question, model_name, temperature):
    llm = Ollama(model=model_name, temperature=temperature)  
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Title of the app
st.title("Enhanced Q&A Chatbot With Ollama")

# Sidebar options
llm = st.sidebar.selectbox("Select Open Source model", ["llama3.2:1b"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# Main user input interface
st.write("Go ahead and ask any question")

with st.form("chat_form"):
    user_input = st.text_input("You:")
    submit_button = st.form_submit_button("Submit")

if submit_button and user_input:
    response = generate_response(user_input, llm, temperature)  
    st.write(response)
else:
    st.write("Please provide user input.")
