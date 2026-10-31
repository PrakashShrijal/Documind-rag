# 📚 DocuMind — AI-Powered Document Assistant

An AI-powered document question-answering system built using **Retrieval-Augmented Generation (RAG)**. DocuMind allows users to upload PDF documents, extract and process their content, and ask questions to receive context-aware answers generated using Google's Gemini API.

## 🚀 Project Overview

Reading and searching through lengthy PDF documents can be time-consuming. DocuMind simplifies this process by allowing users to interact with their documents through natural-language questions.

The application uses semantic search to retrieve relevant information from uploaded PDFs and passes the retrieved context to a Large Language Model (LLM) to generate answers.

## ✨ Key Features

- **PDF Upload:** Upload PDF documents through an interactive Streamlit interface.
- **Text Extraction:** Extract text from PDF pages using PyPDF.
- **Text Chunking:** Divide extracted content into smaller overlapping chunks.
- **Semantic Embeddings:** Convert text chunks into numerical vectors using Sentence Transformers.
- **Vector Database:** Store and manage embeddings using ChromaDB.
- **Semantic Search:** Retrieve relevant document passages based on user questions.
- **AI-Powered Answers:** Generate context-aware responses using Google's Gemini API.
- **Source References:** Display page numbers associated with retrieved document passages.

## 🛠️ Tech Stack

| Technology                   | Purpose |
|---                           |---|
| Python                       | Core programming language |
| Streamlit                    | Interactive web application |
| PyPDF                        | PDF text extraction |
| Sentence Transformers        | Text embedding generation |
| all-MiniLM-L6-v2             | Pretrained embedding model |
| ChromaDB                     | Vector database and similarity search |
| Google Gemini API            | AI-powered answer generation |
| python-dotenv                | Secure environment variable management |

## 🏗️ RAG Architecture

DocuMind follows a Retrieval-Augmented Generation pipeline.

### 1. Document Ingestion Pipeline

1. User uploads a PDF document.
2. PyPDF extracts text from individual pages.
3. Extracted text is divided into chunks of 800 characters with 150 characters of overlap.
4. Sentence Transformers converts each chunk into a numerical embedding.
5. ChromaDB stores the embeddings, text, and page metadata.

### 2. Question-Answering Pipeline

1. User enters a question.
2. The question is converted into an embedding using the same embedding model.
3. ChromaDB performs semantic similarity search.
4. The most relevant document chunks are retrieved.
5. Retrieved chunks are passed as context to Google's Gemini API.
6. Gemini generates an answer based on the supplied document context.
7. The application displays the answer and retrieved source page references.

## 🔄 System Workflow

```text
              PDF Upload
                  |
                  v
           Text Extraction
                  |
                  v
             Chunking
                  |
                  v
          Embedding Generation
                  |
                  v
              ChromaDB
                  |
                  |
       User Enters a Question
                  |
                  v
         Question Embedding
                  |
                  v
        Semantic Similarity Search
                  |
                  v
         Retrieve Relevant Chunks
                  |
                  v
          Gemini API (LLM)
                  |
                  v
          Contextual Answer
                  |
                  v
         Source Page References
```

## 📂 Project Structure

```text
documind-rag/
│
├── app.py                 # Streamlit application
├── rag_pipeline.py        # RAG pipeline implementation
├── requirements.txt       # Project dependencies
├── .env                   # Gemini API key
├── .gitignore             # Git ignored files
├── README.md              # Project documentation
│
├── .chroma/               # Persistent vector database
└── venv/                  # Python virtual environment
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- Google Gemini API key
- VS Code or another Python IDE

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd documind-rag
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Configure Gemini API Key

Create a `.env` file in the project root directory.

Add your API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Important:** Never upload your `.env` file or expose your API key publicly.

### Step 5: Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🧠 Concepts Demonstrated

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Text preprocessing and chunking
- Semantic embeddings
- Vector databases
- Similarity search
- Prompt engineering
- Context-grounded AI responses
- REST/API integration concepts

## 🔮 Future Improvements

- Support multiple PDF documents with document-level isolation.
- Improve chunking using structure-aware or semantic techniques.
- Add more accurate source citations for individual answer claims.
- Implement reranking to improve retrieval relevance.
- Add conversation history and multi-turn question answering.
- Add OCR support for scanned PDF documents.
- Improve the interface with a conversational chatbot experience.
- Add document management and deletion functionality.

## 👨‍💻 Author

**Prakash Shrijal**

⭐ If you find this project interesting, consider giving the repository a star!