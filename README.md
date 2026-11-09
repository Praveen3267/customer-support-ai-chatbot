# Customer Support AI Chatbot

A personal AI-powered customer support chatbot built using Python, Retrieval-Augmented Generation (RAG), and Natural Language Processing (NLP).

## Technologies Used

- Python
- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Supabase with pgvector
- Groq API (`openai/gpt-oss-20b`)
- Retrieval-Augmented Generation (RAG)

## Features

- Answers customer support questions.
- Retrieves relevant information from a knowledge base.
- Uses semantic search to find relevant answers.
- Connects to Supabase for vector storage.
- Uses Groq for AI-generated responses.
- Provides an interactive Streamlit interface.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your API credentials in `api.env`.

Run the application:

```bash
python -m streamlit run rag_app.py
```

## Environment Variables

The application requires:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `GROQ_API_KEY`

Keep your API credentials private. Do not upload `api.env` to GitHub.

## Project Status

Cloud integration with Supabase and Groq completed. Deployment preparation in progress.m