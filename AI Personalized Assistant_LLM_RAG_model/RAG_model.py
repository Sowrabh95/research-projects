#Install the required libraries
import streamlit as st
from dotenv import load_dotenv
import fitz  # pymupdf
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings


#Install Ollama in your local machine from https://ollama.com/download
#RUN COMMAND IN TERMINAL streamlit run RAG_model.py and connect to your local host

# Load environment variables
load_dotenv()

# Streamlit App
st.title('Machine assistant')
st.subheader("Upload a PDF and ask questions about its content.")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save the uploaded PDF locally
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Function to extract text using pymupdf (better for UniJIS-UTF16-H encoding)
    def extract_text_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])  # Extracts text properly
        return text

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf("temp.pdf")

    # Split text into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([pdf_text])  # Creates document chunks

    # Generate embeddings & store in FAISS index
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create Retriever
    retriever = vectorstore.as_retriever()

    st.success("PDF processed successfully! You can now ask questions.")

    # User Input
    input_text = st.text_input("Enter your query:")

# Retrieve relevant chunks
    if input_text:
        retrieved_docs = retriever.get_relevant_documents(input_text)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Define the Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the query based on the provided PDF content."),
            ("user", "Context:\n{context}\n\n Question: {question}")
        ])

        # Initialize LLM
        llm = Ollama(model="llama3")  

        # Define the chain
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        # Generate Answer
        response = chain.invoke({'context': context, 'question': input_text})
        st.write(response)
