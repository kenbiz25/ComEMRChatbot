ComEMR Chatbot
An AI-powered support assistant for ComEMR, built with FastAPI (backend) and React + Vite (frontend). The chatbot uses Retrieval-Augmented Generation (RAG) to answer queries from a local knowledge base and integrates with OpenAI GPT models.

📂 Project Structure
ComEMRChatbot/
├── Backend/                # FastAPI backend
│   ├── main.py             # API routes and chatbot logic
│   ├── indexer.py          # KB indexing script
│   ├── requirements.txt    # Python dependencies
│   ├── data/               # Sessions, KB chunks, FAISS index
│   └── kb/                 # Knowledge base documents (.docx, .pdf)
├── Frontend/               # React + Vite frontend
│   ├── src/                # Components and UI logic
│   ├── package.json        # Node dependencies
│   └── vite.config.js      # Vite configuration
└── README.md               # Project documentation

Features

FastAPI backend with REST endpoints for chat and KB management.
React + Vite frontend with modern UI (Copilot-style).
RAG pipeline using FAISS for semantic search.
KB ingestion from DOCX, PDF, and text files.
OpenAI GPT integration for natural language responses.
Configurable chunking for better retrieval performance.

Prerequisites

Python 3.10+
Node.js 18+
Git
(Optional) Git LFS for large files

Setup Instructions
Backend
Shellcd Backendpython -m venv .venv.\.venv\Scripts\activate   # Windowspip install -r requirements.txt# Run FastAPI serveruvicorn main:app --reload --host 0.0.0.0 --port 8000Show more lines
Frontend
Shellcd Frontendnpm installnpm run devShow more lines

KB Indexing
To index documents for RAG:
Shellcd BackendcdShow more lines
Flags:

--kb → Path to KB folder
--namespace → Index namespace
--chunk-size → Tokens per chunk (recommended: 500–800)
--overlap → Overlap between chunks
--model → Embedding model
--rebuild → Force rebuild of index

Environment Variables
Create a .env file in Backend/:

(Ensure .env is in .gitignore)

Roadmap

✅ Backend + Frontend integration
✅ KB ingestion and indexing
🔲 WhatsApp gateway integration
🔲 Role-based responses (CHW, Admin)
🔲 Deployment (Docker + CI/CD)
🔲 Multi-language support (English + Swahili)

Contributing

Fork the repo
Create a feature branch:
Shellgit checkout -b feature/new-featureShow more lines

Commit changes:
Shellgit commit -m "Add new feature"Show more lines

Push and open a PR.