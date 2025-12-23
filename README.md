
# ComEMR AI Powered Chatbot  
An **AI-powered support assistant** for **ComEMR**, built with **FastAPI (backend)** and **React + Vite (frontend)**.  
The chatbot uses **Retrieval-Augmented Generation (RAG)** to answer queries from a local knowledge base and integrates with **OpenAI GPT models**.

---

## 📂 Project Structure

```
    ComEMRChatbot/
1.  ├── Backend/                # FastAPI backend
2.  │   ├── main.py             # API routes and chatbot logic
3.  │   ├── indexer.py          # KB indexing script
4.  │   ├── requirements.txt    # Python dependencies
5.  │   ├── data/               # Sessions, KB chunks, FAISS index
6.  │   └── kb/                 # Knowledge base documents (.docx, .pdf)
7.  ├── Frontend/               # React + Vite frontend
8.  │   ├── src/                # Components and UI logic
9.  │   ├── package.json        # Node dependencies
10. │   └── vite.config.js      # Vite configuration
11. └── README.md               # Project documentation
```

---

## Features
- **FastAPI backend** with REST endpoints for chat and KB management  
- **React + Vite frontend** with modern UI (Copilot-style)  
- **RAG pipeline** using FAISS for semantic search  
- KB ingestion from **DOCX, PDF, and text files**  
- **OpenAI GPT integration** for natural language responses  
- Configurable **chunking** for better retrieval performance  

---

## Prerequisites
- Python **3.10+**  
- Node.js **18+**  
- Git  
- *(Optional)* Git LFS for large files  

---

## Setup Instructions

### Backend
```bash
cd Backend
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
# Run FastAPI server
```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
### Frontend
```bash
cd Frontend
npm install
npm run dev
```

## KB Indexing
To index documents for RAG:
```
Shellcd Backendpython indexer.py --kb ./kb --namespace default --chunk-size 600 --overlap 80 --model all-MiniLM-L6-v2 --rebuild trueShow more lines
```

Flags:
```
--kb → Path to KB folder
--namespace → Index namespace
--chunk-size → Tokens per chunk (recommended: 500–800)
--overlap → Overlap between chunks
--model → Embedding model
--rebuild → Force rebuild of index
```

## Environment Variables
Create a .env file in Backend/
(Ensure .env is in .gitignore)

## Roadmap
```
✅ Backend + Frontend integration
✅ KB ingestion and indexing
✅  Role-based responses (CHW, Admin)
🔲 WhatsApp gateway integration 
🔲 Deployment (Docker + CI/CD)
🔲 Multi-language support (English + Kreo)
```

## Contributing

1. Fork the repo
2. Create a feature branch
```
git checkout -b feature/new-featureShow more lines
```
3. Commit changes
```
git commit -m "Add new feature"Show more lines
```
4. Push and open a PR


