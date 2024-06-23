from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
import openai
import pandas as pd
import os
from dotenv import load_dotenv


# Get the API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

@st.cache_resource
def load_files(file_paths):
    loaders = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loaders.append(PyPDFLoader(file_path))
        elif file_path.endswith('.txt'):
            loaders.append(TextLoader(file_path, encoding='utf-8'))
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
            text_data = df.to_string(index=False)
            text_file_path = file_path.replace('.csv', '.txt')
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(text_data)
            loaders.append(TextLoader(text_file_path, encoding='utf-8'))
        else:
            st.warning(f"Unsupported file format: {file_path}")

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    ).from_loaders(loaders)
    return index

# Example file paths
file_paths = [
    'H6_User Manual_EN(EU)_V1.0.0.pdf',
    'data.txt',
    'intents-responses-01.csv',
    'BM1(Bear)_User Manual_EN(EU)(V1.0.0).pdf',
    'H6c_QSG_EN(EU)(V1.0.0).pdf',
    'H8c(4G)_User Manual_EN(EU)231027.pdf',
    'intents-responses-01.txt',
    'M3000_V1.0_Datasheet.pdf',
    'Regulatory_Compliance_LT12.pdf',
    'Regulatory_Compliance_LT18.pdf',
    'TY1_User Manual_EN(V1.0.0).pdf',
    'TY2_User Manual_EN(V1.0.0).pdf'
    # Add more file paths as needed
]

# Load the files and create the index
index = load_files(file_paths)

def get_response_from_openai(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response['choices'][0]['message']['content']

st.title('Security Help Desk!')
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'direct_messages' not in st.session_state:
    st.session_state.direct_messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Get user input
prompt = st.chat_input('Pass your prompt here')

if prompt:
    # Display user message
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })
    
    # Retrieve relevant information from the files using the retriever
    docs = index.vectorstore.as_retriever().get_relevant_documents(prompt)
    context = " ".join([doc.page_content for doc in docs])
    
    # Add context to messages for GPT-4
    messages = st.session_state.messages + [{"role": "assistant", "content": f"Context: {context}"}]

    # Get response from OpenAI for the main chatbot
    response = get_response_from_openai(messages)
    
    # Display assistant message
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response
    })

    # Get direct response from GPT-4 with history
    st.session_state.direct_messages.append({
        'role': 'user',
        'content': prompt
    })
    direct_response = get_response_from_openai(st.session_state.direct_messages)
    st.session_state.direct_messages.append({
        'role': 'assistant',
        'content': direct_response
    })
    
    # Display the direct GPT-4 response in the sidebar
    with st.sidebar:
        st.title('Direct GPT-4 Response')
        st.markdown(direct_response)
