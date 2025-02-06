import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings as embed
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# Show title and description.
st.title(f" DUMBO{u"\U0001F914"}")
st.write(
   "💬This chatbot uses Retrieval-Augmented Generation (RAG) with Whisper for audio transcription and Google Gemini for response generation.\nAs of now the Chatbot can only read text from PDF,so avoid using other files to avoid error"
    )


api_key= "AIzaSyDKVNSdigi4vloK1qhqq1rD9gEBFBT6w_w"


# defining a function that reads pdfs
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def split_to_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks=text_splitter.split_text(text)
    return chunks

def vector_stores(text_chunks,api_key):
    embeddings=embed(model="models/embedding-001",google_api_key=api_key)
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def get_conversational_chain(retriever, api_key):
    prompt_template = """
   1)Answer the question as detailed as possible from the provided context atleast 100 words including all the information\n
    
   
    2)If the exact answer is not found in the documents,expand on relevant topics using your creativity\n
    3)If the answer is not in the context:
    - Answer without relying on the context.
    - Start the answer with: "Answering in general as context is not provided:"\n\n

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest", 
        temperature=0.3, 
        google_api_key=api_key
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def user_input(user_question, api_key):
    # Load embeddings and FAISS index
    embeddings = embed(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Create retriever from FAISS index
    retriever = new_db.as_retriever(search_kwargs={"k": 8},similarity_score_threshold=0.7)
    
    # Get QA chain
    qa_chain = get_conversational_chain(retriever, api_key)
    
    # Get response
    response = qa_chain({"query": user_question})
    
    # Display response
    st.write("Reply:\n", response["result"])
    

# modelling on streamlit
def main():
    st.header("AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
    
    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = split_to_chunks(raw_text)
                vector_stores(text_chunks, api_key)
                st.success("Done")

if __name__ == "__main__":
    main()