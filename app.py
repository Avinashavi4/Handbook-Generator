"""
📖 Handbook Generator — AI-Powered Chat Application
=====================================================
Main Gradio application that ties together:
  • PDF Upload & Processing
  • LightRAG Knowledge Graph + Supabase Vector Store
  • Chat Interface with Contextual Responses
  • 20,000-Word Handbook Generation (AgentWrite technique + Grok 4.1)

Built for the LunarTech AI Engineering Assignment.
"""

import os
import shutil
import logging
import tempfile
from typing import List

import gradio as gr

from config import UPLOAD_DIR, XAI_API_KEY, SUPABASE_URL
from pdf_processor import process_pdf, process_multiple_pdfs
from rag_engine import RAGEngine
from handbook_generator import (
    chat_response,
    generate_handbook,
    count_words,
)

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Global State ────────────────────────────────────────────
rag_engine = RAGEngine()


# ─────────────────────────────────────────────────────────────
#  PDF Upload Handler
# ─────────────────────────────────────────────────────────────

def handle_pdf_upload(files) -> str:
    """Process uploaded PDF files and ingest into RAG system."""
    if not files:
        return "⚠️ No files uploaded. Please upload one or more PDF files."

    results = []
    for filepath in files:
        try:
            filename = os.path.basename(filepath)
            dest_path = os.path.join(UPLOAD_DIR, filename)
            shutil.copy2(filepath, dest_path)

            # Process PDF
            full_text, chunks = process_pdf(dest_path)

            # Ingest into RAG
            rag_engine.ingest_document(full_text, chunks, source=filename)

            word_count = count_words(full_text)
            results.append(f"✅ **{filename}** — {word_count:,} words, {len(chunks)} chunks indexed")
            logger.info(f"Processed {filename}: {word_count} words, {len(chunks)} chunks")

        except Exception as e:
            results.append(f"❌ **{os.path.basename(filepath)}** — Error: {str(e)}")
            logger.error(f"Failed to process {filepath}: {e}")

    docs = rag_engine.get_document_list()
    summary = "\n".join(results)
    return f"""## 📄 Upload Results

{summary}

---
**Total documents indexed:** {len(docs)}
**Documents:** {', '.join(docs)}

You can now:
- 💬 **Chat** — Ask questions about your documents
- 📖 **Generate Handbook** — Request a 20,000-word handbook on any topic from your documents
"""


# ─────────────────────────────────────────────────────────────
#  Chat Handler
# ─────────────────────────────────────────────────────────────

HANDBOOK_KEYWORDS = [
    "generate a handbook", "create a handbook", "write a handbook",
    "generate handbook", "create handbook", "write handbook",
    "20000 word", "20,000 word", "20k word",
    "comprehensive handbook", "full handbook",
    "generate a guide", "create a guide", "write a guide",
    "generate a manual", "create a manual",
]


def is_handbook_request(message: str) -> bool:
    """Detect if the user is requesting a handbook generation."""
    message_lower = message.lower()
    return any(kw in message_lower for kw in HANDBOOK_KEYWORDS)


def respond(message: str, history: list) -> str:
    """
    Main chat handler for Gradio 6 messages-format Chatbot.
    `history` is a list of {"role": "user"/"assistant", "content": "..."} dicts.
    Returns the assistant's response string.
    """
    if not rag_engine.get_document_list():
        return ("📄 **No documents uploaded yet!**\n\n"
                "Please upload PDF documents first using the upload panel on the left, "
                "then you can chat about them or request a handbook.")

    # Regular chat — get context and respond
    context = rag_engine.get_context(message)

    # Build chat history for the LLM (last 3 exchanges)
    recent_history = []
    for msg in (history or [])[-6:]:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            recent_history.append(msg)

    return chat_response(message, context, recent_history)


def respond_streaming(message: str, history: list):
    """
    Streaming chat handler — yields partial responses like ChatGPT.
    For handbook requests, streams section by section.
    For regular chat, streams token by token.
    """
    if not rag_engine.get_document_list():
        yield ("📄 **No documents uploaded yet!**\n\n"
               "Please upload PDF documents first using the upload panel on the left, "
               "then you can chat about them or request a handbook.")
        return

    if is_handbook_request(message):
        # Stream handbook generation section by section
        full_context = rag_engine.get_full_context()
        specific_context = rag_engine.get_context(message, top_k=15)
        combined_context = f"{specific_context}\n\n{full_context[:2000]}"

        accumulated = ""
        for part in generate_handbook(message, combined_context):
            accumulated += part
            yield accumulated
        return

    # Regular chat — stream token by token
    context = rag_engine.get_context(message)
    recent_history = []
    for msg in (history or [])[-6:]:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            recent_history.append(msg)

    # Use streaming LLM call
    from handbook_generator import get_client, XAI_MODEL
    system_prompt = ("You are a knowledgeable AI assistant. Answer questions about "
                     "the uploaded documents using the provided context. Be helpful and concise.")
    user_prompt = f"Context:\n{context[:3000]}\n\nQuestion: {message}"

    client = get_client()
    try:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_prompt})

        stream = client.chat.completions.create(
            model=XAI_MODEL,
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            stream=True,
        )
        accumulated = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                accumulated += chunk.choices[0].delta.content
                yield accumulated
    except Exception as e:
        yield f"Error: {e}\n\nPlease check your API key."


# ─────────────────────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(title="📖 AI Handbook Generator") as app:

        # Header
        gr.Markdown(
            "# 📖 AI Handbook Generator\n"
            "### Upload PDFs → Chat → Generate 20,000-Word Handbooks\n"
            "*Powered by Grok 4.1 + LightRAG + Supabase | AgentWrite Technique*\n\n---"
        )

        with gr.Row():
            # ──── Left Panel — Upload & Status ────────────────
            with gr.Column(scale=1):
                gr.Markdown("## 📄 Document Upload")

                file_upload = gr.File(
                    label="Upload PDF Documents",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                )

                upload_btn = gr.Button(
                    "🚀 Process & Index PDFs", variant="primary", size="lg"
                )

                upload_status = gr.Markdown(value=_get_status_text())

                gr.Markdown(
                    "---\n"
                    "### 💡 Quick Commands\n"
                    "- **Chat**: Ask any question about your docs\n"
                    '- **Handbook**: *"Create a handbook on [topic]"*\n'
                    '- **Guide**: *"Generate a comprehensive guide about..."*\n\n'
                    "### ⚙️ System Info"
                )

                gr.Markdown(value=_get_system_info())

            # ──── Right Panel — Chat ──────────────────────────
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 Chat & Handbook Generation",
                    height=500,
                    placeholder="Upload PDFs, then ask questions or request a handbook!",
                    render_markdown=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder='Try: "Create a handbook on Retrieval-Augmented Generation"',
                        scale=4,
                        lines=2,
                    )
                    send_btn = gr.Button("Send 📨", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat")
                    export_btn = gr.Button("📥 Export Last Response")

                export_output = gr.File(label="Exported Handbook", visible=False)

        # ─── Event Handlers ──────────────────────────────────

        def on_upload(files):
            """Handle PDF upload button click."""
            if files is None:
                return "⚠️ No files selected."
            return handle_pdf_upload(files)

        def on_send(message: str, history: list):
            """Streaming handler — yields progressive updates like ChatGPT."""
            if not message.strip():
                yield "", history
                return

            history = history or []
            history.append({"role": "user", "content": message})

            # Add empty assistant message that we'll fill progressively
            history.append({"role": "assistant", "content": ""})

            # Stream the response
            for partial_response in respond_streaming(message, history[:-2]):
                history[-1]["content"] = partial_response
                yield "", history

        def on_clear():
            """Clear the chatbot and export output."""
            return [], gr.update(visible=False)

        def on_export(history: list):
            """Export the last assistant response as a Markdown file."""
            if not history:
                return gr.update(visible=False)

            # Find last assistant message
            last_content = ""
            for msg in reversed(history):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    last_content = msg["content"]
                    break

            if not last_content:
                return gr.update(visible=False)

            export_path = os.path.join(tempfile.gettempdir(), "handbook_export.md")
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(last_content)
            return gr.update(value=export_path, visible=True)

        # Wire up events
        upload_btn.click(fn=on_upload, inputs=[file_upload], outputs=[upload_status])

        send_btn.click(
            fn=on_send, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot]
        )
        msg_input.submit(
            fn=on_send, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot]
        )

        clear_btn.click(fn=on_clear, outputs=[chatbot, export_output])
        export_btn.click(fn=on_export, inputs=[chatbot], outputs=[export_output])

    return app


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _get_status_text() -> str:
    docs = rag_engine.get_document_list()
    if docs:
        return f"✅ **{len(docs)} document(s) loaded:** {', '.join(docs)}"
    return "📭 No documents uploaded yet. Upload PDFs to get started."


def _get_system_info() -> str:
    api_ok = "✅ Configured" if XAI_API_KEY and XAI_API_KEY not in ("your_xai_api_key_here", "your_api_key_here") else "❌ Not set"
    supa_ok = "✅ Configured" if SUPABASE_URL else "⚠️ Local storage"
    return (
        "| Component | Status |\n"
        "|-----------|--------|\n"
        f"| **Grok API** | {api_ok} |\n"
        f"| **Supabase** | {supa_ok} |\n"
        "| **LightRAG** | Auto-detected |\n"
        "| **PDF Engine** | ✅ Ready |"
    )


# ─────────────────────────────────────────────────────────────
#  Main Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Handbook Generator...")
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
