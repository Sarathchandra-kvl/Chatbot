
# 🧠 DUMBO: Audio-Enhanced RAG Chatbot using Google Gemini

DUMBO is an intelligent chatbot built with **Retrieval-Augmented Generation (RAG)** powered by **Google Gemini**, enhanced with **Whisper for audio transcription** and **gTTS for voice output**. It allows users to ask questions about uploaded PDF documents via voice or text and responds in both text and audio formats.

<!-- ![DUMBO Chatbot Demo](https://your-demo-link.gif) --> <!-- (Optional) Add a demo GIF -->

---

## 🚀 Features

- 📄 **PDF-Based Question Answering** – Upload multiple PDFs and ask questions based on their content.  
- 🎙️ **Voice Input** – Ask your question by speaking; Whisper automatically transcribes it.  
- 🔊 **Voice Output** – Get answers read aloud using Google Text-to-Speech (gTTS).  
- 🧠 **Google Gemini Integration** – Leverages Gemini 1.5 Pro for accurate, context-rich answers.  
- 🗂️ **FAISS Vector Store** – Efficient chunking and retrieval of document data.  
- 🗣️ **Conversational Memory** – Tracks chat history for a natural conversation experience.  

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – UI/UX  
- [Whisper](https://github.com/openai/whisper) – Speech-to-text  
- [gTTS](https://pypi.org/project/gTTS/) – Text-to-speech  
- [Google Gemini Pro](https://ai.google.dev/) – LLM for response generation  
- [LangChain](https://www.langchain.com/) – Chains & RAG pipeline  
- [FAISS](https://github.com/facebookresearch/faiss) – Document embedding store  

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/dumbo-chatbot.git
cd dumbo-chatbot
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Add your **Google Generative AI API Key** in the code (replace placeholder in `api_key = ""`):
```python
api_key = "YOUR_GOOGLE_API_KEY"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📋 How to Use

1. Upload your PDF(s) from the **Sidebar Menu**.  
2. Click **"Submit & Process"** to chunk and index the documents.  
3. Use the **text input** or **record audio** to ask questions.  
4. The assistant will display and read the answer aloud.  
5. Clear chat history anytime via the sidebar.  

---

## 📌 Example Use Cases

- Studying large textbooks  
- Summarizing legal documents  
- Navigating user manuals  
- Quick Q&A from technical PDFs  

---

## 🧩 To-Do / Improvements

- ⏳ Support for longer audio inputs  
- 💾 Save conversation history  
- 🌍 Multilingual support  
- 🔐 Secure API key via environment variables  

---

## 🤝 Contributing

Feel free to submit pull requests or open issues for bugs and feature suggestions.

---

## 📄 License

This project is licensed under the MIT License.
