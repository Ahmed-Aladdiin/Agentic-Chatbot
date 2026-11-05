# Agentic Chatbot

This project is a lightweight agentic chatbot built with Python, Streamlit, LangChain, and LangGraph. It is designed to act like a helpful assistant that can use external tools to answer questions, search the web, calculate values, check weather, fetch stock prices, and answer questions from uploaded PDF documents using retrieval-augmented generation (RAG).

In short, this app is more than a simple chatbot: it can decide when to use tools, keep conversation context, and extend its capabilities beyond plain text generation.

## What this project does

- Chat with an LLM in a simple Streamlit interface
- Use tools for web search, math, weather, and stock data
- Upload PDFs and ask questions from the stored document content
- Maintain multiple conversation threads
- Save chat state and local PDF knowledge bases for each thread

## Tech stack

- Python 3.11+
- Streamlit
- LangChain
- LangGraph
- FAISS vector search
- OpenRouter LLM access
- Google embeddings
- Tavily search

## Project structure

- `main.py` — Streamlit app entry point
- `src/agentic_chatbot/chatbot.py` — LangGraph chatbot setup and thread management
- `src/agentic_chatbot/tools.py` — tools for web search, weather, stock prices, PDF RAG, and math
- `faiss_store/` — vector store created per conversation thread
- `uploaded_pdfs/` — uploaded PDF files for document processing
- `chatbot.db` — local SQLite checkpoint database for chat state

## Requirements

Before running the app, make sure you have:

- Python 3.11 or newer
- `uv` installed
- API keys for the services used by the app

## Set up environment variables

This project expects environment variables in two places:

1. In the project root folder
2. In `src/agentic_chatbot/`

The app calls `load_dotenv()` from both locations, so keep the same values in both `.env` files to avoid missing keys when the app is launched from different directories.

Example root `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
ALPHAVANTAG_API_KEY=your_alpha_vantage_key
GOOGLE_API_KEY=your_google_api_key
```

Example `src/agentic_chatbot/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
ALPHAVANTAG_API_KEY=your_alpha_vantage_key
GOOGLE_API_KEY=your_google_api_key
```

## Install dependencies with uv

From the project root, run:

```bash
uv sync --frozen
```

If the lockfile is not available or needs to be refreshed, use:

```bash
uv sync
```

This installs the dependencies defined in `pyproject.toml` and uses the lockfile to keep versions consistent.

## Run the app

Start the Streamlit app from the root folder:

```bash
uv run streamlit run main.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to use it

- Ask normal questions in the chat box
- Use the built-in tools automatically when the assistant needs them
- Upload a PDF in the chat experience for document-based Q&A
- Continue the same thread to keep context
- Open a new thread if you want a separate conversation history

## Notes

- The app creates a local FAISS vector index inside `faiss_store/<thread_id>/` for PDFs.
- Conversation state is stored in `chatbot.db`.
- Some features depend on external API keys being configured correctly.

## License

This project is for local development and learning use unless stated otherwise by the repository owner.

## Summary

This repository is a practical example of an agentic AI application: a chatbot that can think, decide, call tools, and use documents as additional context. It is a good base for experimenting with tool-using agents, RAG workflows, and LLM-powered applications in a simple and extendable structure.
