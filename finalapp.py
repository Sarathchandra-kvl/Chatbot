import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings as embed
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import whisper
from gtts import gTTS  # For text-to-speech

# ================================
# 1. Chat History Setup (New)
# ================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

# ================================
# 2. Define Text-to-Speech Function (New)
# ================================
def text_to_speech(text, lang='en'):
    """Converts text to speech and returns the path of the temporary audio file."""
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
        tts.save(tts_file.name)
        return tts_file.name

# ================================
# 3. Title and Description
# ================================
st.title(f"DUMBO {u'\U0001F914'}")
st.write(
    "💬 This chatbot uses Retrieval-Augmented Generation (RAG) with Whisper for audio transcription and Google Gemini for response generation. "
    "Currently, the chatbot reads text from PDFs. Avoid uploading other file types to prevent errors."
)

api_key = "AIzaSyDKVNSdigi4vloK1qhqq1rD9gEBFBT6w_w"

# ================================
# 4. PDF Processing Functions
# ================================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_stores(text_chunks, api_key):
    embeddings = embed(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ================================
# 5. Conversational Chain Function
# ================================
def get_conversational_chain(retriever, api_key):
    prompt_template = """
   1) Answer the question as detailed as possible from the provided context (at least 100 words) including all available information.
   2) If the exact answer is not found in the documents, expand on relevant topics using your creativity.
   3) If the answer is not in the context:
      - Answer without relying on the context.
      - Start the answer with: "Answering in general as context is not provided:"
      
    Context:
    {context}
    
    Question:
    {question}
    
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

# ================================
# 6. User Input Handling Function
# ================================
def user_input(user_question, api_key):
    # Load embeddings and FAISS index
    embeddings = embed(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Create retriever from FAISS index
    retriever = new_db.as_retriever(search_kwargs={"k": 8}, similarity_score_threshold=0.7)
    
    # Get QA chain
    qa_chain = get_conversational_chain(retriever, api_key)
    
    # Get response from the QA chain
    response = qa_chain({"query": user_question})
    
    # Update chat history (New)
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": response["result"]})
    
    # Display Chat History (New)
    st.write("### Chat History:")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write("**You:**", msg["content"])
        else:
            st.write("**Bot:**", msg["content"])
    
    # Display the text response
    st.write("### Reply:")
    st.write(response["result"])
    
    # Text-to-Speech: Convert the response to audio and play it (New)
    tts_audio_file = text_to_speech(response["result"])
    with open(tts_audio_file, "rb") as audio_f:
        st.audio(audio_f.read(), format="audio/mp3")
    os.remove(tts_audio_file)

# ================================
# 7. Main Function
# ================================
def main():
    st.header("AI Clone Chatbot 💁")
    
    # --- Speech-to-Text Section (New) ---
    st.subheader("Speak Your Question:")
    # Using Streamlit's new audio_input for recording (returns a file-like object)
    audio_input = st.audio_input("Record your question audio")
    if audio_input is not None:
        with st.spinner("Transcribing audio..."):
            try:
                model = whisper.load_model("base")
                # Save the recorded audio to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    temp_audio.write(audio_input.getbuffer())
                    temp_audio_path = temp_audio.name
                result = model.transcribe(temp_audio_path)
                transcribed_text = result["text"]
                # Pre-fill the text input with the transcribed text
                st.session_state.user_question = transcribed_text
                os.remove(temp_audio_path)
            except Exception as e:
                st.error(f"Error transcribing audio: {e}")
    
    # --- Text Input Section ---
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question", value=st.session_state.user_question)
    
    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)
    
    # --- Sidebar: PDF Upload and Processing Section ---
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = split_to_chunks(raw_text)
                vector_stores(text_chunks, api_key)
                st.success("PDF processing complete!")

if __name__ == "__main__":
    main()
