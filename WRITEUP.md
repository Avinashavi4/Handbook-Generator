# AI Handbook Generator — Technical Write-Up

## 📋 Assignment Summary

This project implements an **AI-Powered Handbook Generator** for the LunarTech AI Engineering Assignment. The system uploads PDF documents, builds a knowledge base, enables conversational Q&A, and generates comprehensive 20,000-word handbooks using the **AgentWrite** technique powered by **Grok 4.1**.

---

## 🏗️ Architecture

```
User → Gradio UI → PDF Processor → RAG Engine → LLM (Grok 4.1)
                                       ↓
                              Knowledge Graph (LightRAG)
                              + Vector Store (Supabase/Local)
```

### Components

| Module | Responsibility |
|--------|----------------|
| **app.py** | Gradio 6 web interface — chat, PDF upload, handbook export |
| **pdf_processor.py** | PDF text extraction (pdfplumber), sentence-aware chunking |
| **rag_engine.py** | LightRAG knowledge graph + Supabase pgvector store with local fallback |
| **handbook_generator.py** | AgentWrite pipeline — planning & iterative section generation |
| **config.py** | Centralized configuration from environment variables |
| **prompts/** | Externalized LLM prompt templates for plan and write phases |

---

## 🔑 Key Design Decisions

### 1. AgentWrite for Long-Form Generation
- Standard LLMs struggle with coherent 20K-word output in a single pass
- **Solution**: Two-phase approach:
  - **Plan Phase**: LLM breaks the topic into 25-40 structured sections with word targets
  - **Write Phase**: Each section generated individually with a rolling context window (last 3,000 words) for coherence
- This ensures consistent quality and allows word-count management per section

### 2. Hybrid RAG (Knowledge Graph + Vector Search)
- **LightRAG**: Builds a knowledge graph from documents for relationship-aware retrieval
- **Supabase pgvector**: Traditional vector similarity search as a complementary retrieval path
- **Local Fallback**: NumPy-based cosine similarity when external services are unavailable
- Results are merged and deduplicated for comprehensive context

### 3. Graceful Degradation
- Every external dependency (Supabase, LightRAG, Grok API) has a fallback path
- The app remains functional even with only local components
- Clear status indicators show which components are active

### 4. Chunking Strategy
- **Sentence-aware chunking**: Splits at sentence boundaries to preserve semantic coherence
- **1,000-word chunks with 200-word overlap**: Balances context size with retrieval granularity
- Metadata preserved for source attribution

---

## 🧰 Technology Stack

| Technology | Purpose | Why Chosen |
|-----------|---------|------------|
| **Grok 4.1** (xAI API) | LLM for generation & chat | Assignment requirement; OpenAI-compatible API |
| **LightRAG** | Knowledge graph construction | Captures entity relationships beyond keyword matching |
| **Supabase + pgvector** | Vector storage & retrieval | Managed Postgres with native vector search |
| **Gradio 6** | Web UI | Rapid prototyping with chat interface support |
| **pdfplumber** | PDF extraction | Reliable text extraction from varied PDF formats |
| **Python 3.11** | Runtime | Modern async support, type hints |

---

## 📊 Prompt Engineering

### Planning Prompt (`prompts/plan.txt`)
- Instructs the LLM to break a topic into numbered sections
- Each section has: title, description, word target, relevant keywords
- Targets 25-40 sections summing to ~20,000 words
- Format is parseable for automated section extraction

### Writing Prompt (`prompts/write.txt`)
- Provides section context: plan overview, previously written text, current section details
- Enforces word count targets per section
- Maintains academic/professional tone
- Includes rolling context window for cross-section coherence

---

## ⚡ Performance Considerations

- **Streaming preparation**: The generate_handbook function yields results incrementally
- **Chunked context**: Only relevant chunks are sent to the LLM, not entire documents
- **Rolling window**: Limits context size while maintaining coherence (last 3,000 words)
- **Parallel-ready**: Architecture supports future async generation of independent sections

---

## 🔄 How It Works (User Flow)

1. **Upload PDFs** → pdfplumber extracts text → chunks created → ingested into RAG engine
2. **Chat** → User question → RAG retrieves relevant context → Grok generates contextual answer
3. **Generate Handbook** → Detected by keyword matching → AgentWrite plan phase → iterative section generation → full handbook returned
4. **Export** → Last response exported as Markdown file

---

## 🚀 Setup & Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. (Optional) Set up Supabase
# Run supabase_setup.sql in your Supabase SQL editor

# 4. Launch
python app.py
# Open http://localhost:7860
```

---

## 📝 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Gradio 6 API breaking changes | Adapted to messages-format Chatbot, updated theme/CSS APIs |
| 20K words exceeds single LLM context | AgentWrite: plan → write sections with rolling context |
| PDF formatting variability | pdfplumber + text cleaning + sentence-aware chunking |
| LLM hallucination risk | RAG grounds responses in actual document content |
| Supabase cold start / unavailability | Local NumPy-based vector store fallback |

---

## 📁 Project Structure

```
handbook-generator/
├── app.py                  # Main Gradio application
├── config.py               # Configuration management
├── pdf_processor.py        # PDF extraction & chunking  
├── rag_engine.py           # LightRAG + Supabase + local fallback
├── handbook_generator.py   # AgentWrite pipeline
├── prompts/
│   ├── plan.txt            # Planning prompt template
│   └── write.txt           # Section writing prompt template
├── supabase_setup.sql      # Database setup script
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── README.md               # User documentation
└── WRITEUP.md              # This technical write-up
```

---

*Built for the LunarTech SilverAI AI Engineering Assignment*
