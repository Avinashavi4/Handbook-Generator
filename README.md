# 📖 AI Handbook Generator

> **AI-Powered Chat Application** that generates **20,000-word handbooks** from uploaded PDF documents.  
> Built with **Grok 4.1** + **LightRAG** + **Supabase** using the **AgentWrite** technique.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?logo=gradio)
![Grok](https://img.shields.io/badge/LLM-Grok%204.1-red)
![License](https://img.shields.io/badge/License-Confidential-yellow)

---

## 🎯 What It Does

1. **📄 Upload PDFs** — Drop in research papers, documentation, any text-based PDFs
2. **🧠 Knowledge Graph** — LightRAG + Supabase create a searchable knowledge graph
3. **💬 Chat** — Ask questions about your documents and get contextual AI responses
4. **📖 Generate Handbooks** — Request a **20,000+ word** structured handbook via chat

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HANDBOOK GENERATOR                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📄 PDF Upload  →  🧠 LightRAG  →  💬 Chat UI  →  📖 Handbook │
│   (pdfplumber)    (Supabase)      (Grok 4.1)    (20k words)│
│                                                             │
└─────────────────────────────────────────────────────────────┘

Core Flow:
  1. User uploads PDFs → pdfplumber extracts and chunks text
  2. Chunks stored in Supabase → LightRAG creates knowledge graph
  3. User chats → Context retrieved from knowledge graph → Grok responds
  4. Handbook request → AgentWrite pipeline → 20,000-word output
```

### Technology Stack

| Component          | Technology           | Purpose                                   |
| ------------------ | -------------------- | ----------------------------------------- |
| **Frontend**       | Gradio               | Chat interface with PDF upload             |
| **LLM**            | Grok 4.1 (xAI API)  | Long-context generation                    |
| **RAG System**     | LightRAG             | Knowledge graph from PDFs                  |
| **Vector Storage** | Supabase (pgvector)  | Embeddings and document storage            |
| **PDF Processing** | pdfplumber           | Text extraction from uploads               |

### AgentWrite Pipeline (for 20k+ word generation)

The handbook generation follows the **AgentWrite** technique from the LongWriter paper:

1. **PLAN** — Break the handbook request into 25-40 structured sections with word count targets
2. **WRITE** — Generate each section sequentially, feeding previous context to maintain coherence
3. **ASSEMBLE** — Combine all sections into a complete, structured handbook

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **xAI API Key** (for Grok 4.1) — Get from [x.ai](https://x.ai)
- **Supabase Account** (free tier works) — [supabase.com](https://supabase.com)

### 1. Clone & Install

```bash
cd "Lunar Assignment/handbook-generator"
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example env file
cp .env.example .env
```

Edit `.env` with your API keys:

```env
XAI_API_KEY=xai-xxxxxxxxxxxxxxxxxxxx
XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL=grok-4-1

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

### 3. Setup Supabase (One-time)

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run the contents of `supabase_setup.sql`
3. Copy your project URL and anon key to `.env`

### 4. Run the Application

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

---

## 📖 Usage Guide

### Upload Documents
1. Click **"Upload PDF Documents"** in the left panel
2. Select one or more PDF files
3. Click **"🚀 Process & Index PDFs"**
4. Wait for indexing confirmation

### Chat with Documents
Type any question in the chat box:
- *"What are the main findings in the paper?"*
- *"Explain the methodology used"*
- *"Summarize the key concepts"*

### Generate a Handbook
Use trigger phrases to generate a 20,000-word handbook:
- *"Create a handbook on Retrieval-Augmented Generation"*
- *"Generate a comprehensive guide about the research methodology"*
- *"Write a handbook covering all the topics in the uploaded documents"*

### Export
Click **"📥 Export Last Response"** to download the handbook as a Markdown file.

---

## 📁 Project Structure

```
handbook-generator/
├── app.py                    # Main Gradio application
├── config.py                 # Configuration and environment variables
├── pdf_processor.py          # PDF text extraction and chunking
├── rag_engine.py             # LightRAG + Supabase RAG engine
├── handbook_generator.py     # AgentWrite handbook generation pipeline
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .env                      # Your API keys (create from .env.example)
├── supabase_setup.sql        # Supabase database setup script
├── prompts/
│   ├── plan.txt              # AgentWrite planning prompt template
│   └── write.txt             # AgentWrite writing prompt template
├── uploads/                  # Uploaded PDF storage (auto-created)
└── lightrag_data/            # LightRAG working directory (auto-created)
```

---

## 🧪 Test Case

**Input:**
- Upload 2-3 AI-related PDFs
- Chat: *"Create a handbook on Retrieval-Augmented Generation"*

**Expected Output:**
- 20,000+ word structured document
- Table of contents
- Sections with proper headings
- Citations from uploaded PDFs

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No API key" error | Check `.env` file has valid `XAI_API_KEY` |
| Supabase connection failed | Verify URL and key; app falls back to local storage |
| PDF extraction empty | PDF may be image-based; use text-based PDFs |
| LightRAG import error | `pip install lightrag-hku`; app falls back to vector search |
| Handbook generation slow | Expected — 25-40 API calls needed for 20k words |

---

## 📝 Approach & Design Decisions

### Why Gradio?
- Fastest path to a working chat UI with file upload
- Built-in Markdown rendering for handbook output
- No frontend build step required

### Why AgentWrite?
- Standard LLMs have output limits (~4k-8k tokens per response)
- AgentWrite breaks generation into planned sections, each within token limits
- Sequential generation with context from previous sections maintains coherence
- This is the exact technique from the LongWriter paper

### Fallback Design
- **Supabase unavailable?** → Falls back to local numpy-based vector store
- **LightRAG not installed?** → Falls back to cosine similarity search
- **Embedding API fails?** → Falls back to hash-based embeddings
- The app is designed to **always work**, with reduced features if services are unavailable

---

## 📧 Submission

**Author:** [Your Name]  
**Email:** [Your Email]  
**Submit to:** tk.lunartech@gmail.com

---

*Built for the LunarTech AI Engineering Assignment — Handbook Generator*
