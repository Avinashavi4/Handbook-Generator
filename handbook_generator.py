"""
Handbook Generator Module
Implements the AgentWrite pipeline adapted for Grok 4.1:
  1. PLAN: Break the handbook request into 25-40 sections (~20,000 words)
  2. WRITE: Generate each section sequentially, building on previous text
"""

import os
import re
import time
import json
import logging
from typing import List, Callable, Optional, Generator

from openai import OpenAI

from config import (
    XAI_API_KEY, XAI_BASE_URL, XAI_MODEL,
    HANDBOOK_TARGET_WORDS, TEMPERATURE, MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    """Create OpenAI-compatible client for Grok."""
    return OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)


def call_llm(prompt: str, max_tokens: int = 4096, temperature: float = TEMPERATURE) -> str:
    """Call Grok 4.1 via the xAI API (OpenAI-compatible)."""
    if not XAI_API_KEY or XAI_API_KEY in ("your_xai_api_key_here", "your_api_key_here"):
        raise RuntimeError(
            "XAI_API_KEY is not configured. Please set it in the .env file."
        )
    client = get_client()
    for attempt in range(MAX_RETRIES + 3):
        try:
            response = client.chat.completions.create(
                model=XAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if "429" in err_str or "rate_limit" in err_str:
                # Rate limited — wait longer and retry
                wait = min(60, 10 * (attempt + 1))
                logger.info(f"Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"LLM call failed after {attempt + 1} attempts: {e}")


def call_llm_streaming(prompt: str, max_tokens: int = 4096, temperature: float = TEMPERATURE) -> Generator[str, None, None]:
    """Call Grok 4.1 with streaming response."""
    client = get_client()
    try:
        stream = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"Streaming LLM call failed: {e}")
        yield f"\n\n[Error: {e}]"


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


# ─────────────────────────────────────────────────────────────
#  Step 1: PLAN — Break handbook into sections
# ─────────────────────────────────────────────────────────────

def create_plan(instruction: str, context: str) -> List[dict]:
    """
    Use AgentWrite planning to break a handbook request into sections.
    Returns list of {"title": ..., "main_point": ..., "word_count": ...}
    """
    # Load planning prompt template
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "plan.txt")
    template = open(prompt_path, encoding="utf-8").read()

    # Truncate context if too long (keep first ~8k chars for planning)
    if len(context) > 8000:
        context = context[:8000] + "\n\n[... additional context available ...]"

    prompt = template.replace("$INST$", instruction).replace("$CONTEXT$", context)

    logger.info("Generating handbook plan...")
    plan_text = call_llm(prompt, max_tokens=4096, temperature=0.5)

    # Parse the plan into structured sections
    sections = parse_plan(plan_text)

    if not sections:
        # Fallback: generate a default plan
        sections = generate_default_plan(instruction)

    logger.info(f"Plan created: {len(sections)} sections, ~{sum(s['word_count'] for s in sections)} target words")
    return sections


def parse_plan(plan_text: str) -> List[dict]:
    """Parse the LLM-generated plan into structured sections."""
    sections = []
    lines = plan_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to parse: "Section N - Title: X - Main Point: Y - Word Count: Z words"
        title_match = re.search(r'Title:\s*(.+?)(?:\s*-\s*Main Point:)', line)
        point_match = re.search(r'Main Point:\s*(.+?)(?:\s*-\s*Word Count:)', line)
        count_match = re.search(r'Word Count:\s*(\d+)', line)

        if title_match and point_match and count_match:
            sections.append({
                "title": title_match.group(1).strip(),
                "main_point": point_match.group(1).strip(),
                "word_count": int(count_match.group(1)),
            })
        elif 'Main Point:' in line and 'Word Count:' in line:
            # Looser parsing
            parts = line.split(' - ')
            title = parts[0].strip() if parts else "Section"
            point = ""
            wc = 600
            for part in parts:
                if 'Main Point:' in part:
                    point = part.split('Main Point:')[1].strip()
                if 'Word Count:' in part:
                    m = re.search(r'(\d+)', part)
                    if m:
                        wc = int(m.group(1))
            sections.append({
                "title": title,
                "main_point": point,
                "word_count": wc,
            })

    return sections


def generate_default_plan(instruction: str) -> List[dict]:
    """Generate a default handbook plan if LLM planning fails."""
    base_sections = [
        ("Table of Contents", "Comprehensive table of contents listing all sections and sub-sections", 500),
        ("Introduction and Overview", "Introduction to the topic, its importance, scope, objectives, and relevance in the modern landscape", 1800),
        ("Background, History and Evolution", "Historical context, evolution, foundational developments, key milestones and pioneering contributions", 2000),
        ("Core Concepts and Fundamental Principles", "Key concepts, definitions, theories, methodologies and fundamental principles explained in depth", 2000),
        ("Architecture, Design and System Components", "System architecture, design patterns, component interactions, data flow and structural considerations", 2000),
        ("Implementation Strategies and Best Practices", "Practical implementation approaches, step-by-step methodologies, industry best practices and coding patterns", 2000),
        ("Tools, Technologies and Ecosystem", "Comprehensive overview of tools, frameworks, libraries, platforms and community ecosystem", 1800),
        ("Case Studies and Real-World Applications", "Detailed case studies, real-world examples, success stories and practical applications across industries", 1800),
        ("Advanced Topics and Performance Optimization", "Advanced techniques, performance tuning, scaling strategies, benchmarks and optimization patterns", 1800),
        ("Security, Testing and Quality Assurance", "Security best practices, threat modeling, testing methodologies, QA frameworks and validation strategies", 1500),
        ("Deployment, Monitoring and Maintenance", "Deployment strategies, CI/CD pipelines, monitoring tools, troubleshooting guides and maintenance procedures", 1500),
        ("Future Directions and Emerging Trends", "Future outlook, emerging trends, research frontiers, strategic recommendations and industry predictions", 1300),
        ("Glossary, References and Conclusion", "Comprehensive glossary of terms, curated references, additional resources and concluding summary", 1000),
    ]

    return [
        {"title": title, "main_point": f"{point} related to: {instruction}", "word_count": wc}
        for title, point, wc in base_sections
    ]


# ─────────────────────────────────────────────────────────────
#  Step 2: WRITE — Generate each section
# ─────────────────────────────────────────────────────────────

def generate_handbook(
    instruction: str,
    context: str,
    progress_callback: Optional[Callable] = None,
) -> Generator[str, None, None]:
    """
    Generate a complete 20,000-word handbook using the AgentWrite pipeline.
    Yields text progressively (section by section).
    """
    # Step 1: Create the plan
    if progress_callback:
        progress_callback("📋 Creating handbook outline...")

    yield "# Generating Handbook Plan...\n\n"

    sections = create_plan(instruction, context)

    # Build plan text for the write prompts
    plan_text = "\n".join([
        f"Section {i+1} - Title: {s['title']} - Main Point: {s['main_point']} - Word Count: {s['word_count']} words"
        for i, s in enumerate(sections)
    ])

    total_sections = len(sections)
    total_words = 0
    written_text = ""

    # Clear initial message and start actual content
    yield f"\n📖 **Handbook Plan: {total_sections} sections, ~{sum(s['word_count'] for s in sections):,} target words**\n\n---\n\n"

    # Load write prompt template
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "write.txt")
    write_template = open(prompt_path, encoding="utf-8").read()

    # Step 2: Write each section
    for i, section in enumerate(sections):
        section_label = f"Section {i+1} - Title: {section['title']} - Main Point: {section['main_point']} - Word Count: {section['word_count']} words"

        if progress_callback:
            progress_callback(f"✍️ Writing section {i+1}/{total_sections}: {section['title']}...")

        # Build the prompt
        # Truncate already-written text to last ~1500 words for context
        recent_text = _get_recent_text(written_text, max_words=1500)
        section_context = context[:8000] if len(context) > 8000 else context

        prompt = (
            write_template
            .replace("$INST$", instruction)
            .replace("$CONTEXT$", section_context)
            .replace("$PLAN$", plan_text)
            .replace("$TEXT$", recent_text)
            .replace("$STEP$", section_label)
        )

        # Small delay between sections to respect rate limits
        if i > 0:
            time.sleep(0.5)

        # Generate this section
        try:
            section_text = call_llm(prompt, max_tokens=4096, temperature=TEMPERATURE)

            if section_text:
                written_text += section_text + "\n\n"
                total_words += count_words(section_text)
                yield section_text + "\n\n"
                yield f"*[Progress: Section {i+1}/{total_sections} complete | Words: ~{total_words:,}]*\n\n---\n\n"
            else:
                yield f"\n\n*[Section {i+1} generation failed, continuing...]*\n\n"

        except Exception as e:
            logger.error(f"Failed to generate section {i+1}: {e}")
            yield f"\n\n*[Error generating section {i+1}: {e}]*\n\n"

    # Final summary
    yield f"\n\n---\n\n**✅ Handbook Complete!**\n- Total sections: {total_sections}\n- Approximate word count: ~{total_words:,}\n- Target: {HANDBOOK_TARGET_WORDS:,} words\n"


def _get_recent_text(text: str, max_words: int = 3000) -> str:
    """Get the most recent portion of text (last N words)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return "... " + " ".join(words[-max_words:])


# ─────────────────────────────────────────────────────────────
#  Chat Response (non-handbook)
# ─────────────────────────────────────────────────────────────

def chat_response(message: str, context: str, history: List[dict] = None) -> str:
    """Generate a chat response using retrieved context."""
    system_prompt = """You are a knowledgeable AI assistant for the Handbook Generator application. 
You help users understand their uploaded documents and can answer questions about the content.
When answering, reference specific information from the provided context.
If the context doesn't contain relevant information, say so honestly.
Be helpful, clear, and concise."""

    user_prompt = f"""Context from uploaded documents:
{context[:8000]}

User question: {message}

Please provide a helpful, detailed response based on the context above."""

    client = get_client()
    try:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for h in history[-6:]:  # Keep last 3 exchanges
                messages.append(h)
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model=XAI_MODEL,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}\n\nPlease check your API key configuration."
