import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
import pinecone
from io import BytesIO
import os 

import tempfile
from langchain.document_loaders import PyPDFLoader

def process_pdf(uploaded_file):
    """Processes a PDF file and splits it into text chunks."""
    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load the file using the file path
    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    # Optionally delete the file after processing
    os.remove(tmp_file_path)

    return text_chunks


def process_text(file):
    """Processes a text file and splits it into text chunks

    Args:
        file: A file object containing the text data

    Returns:
        A list of text chunks
    """
    loader = TextLoader(file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks


st.title("PDF and Text Query Answering System")

# Set API keys as environment variables (replace with your own)
os.environ['PINECONE_API_KEY'] ="pcsk_2e7k3w_SpifERwQi469twi6PMxVcH9vK8QvTZ3Ch9gyKmdYFTHGvvahDSEUy2GonZihwyp" # available at app.pinecone.io
os.environ['GOOGLE_API_KEY'] ="AIzaSyCKkDF3rElx7D7MBqAlrSO-zVVhatZ8WCg"
os.environ["PINECONE_API_ENV"] = "aws-starter"


uploaded_file = st.file_uploader("Upload your PDF or Text File ", type=['pdf', 'text'])


if uploaded_file is not None:
    # Process uploaded file based on its type
    if ".pdf" in uploaded_file.name:
        st.write("Processing PDF...")
        text_chunks = process_pdf(uploaded_file)
    elif ".txt" in uploaded_file.name:
        st.write("Processing Text file...")
        text_chunks = process_text(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF or text file.")
        text_chunks = None  # Avoid potential errors later

    if text_chunks:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_API_ENV'])

        index_name = 'text'
        dosearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

        Query = st.text_input("Enter Your Query Here")
        docs = dosearch.similarity_search(Query)

        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=dosearch.as_retriever())

        result = qa.invoke(Query)

        if Query:
            st.write(result)