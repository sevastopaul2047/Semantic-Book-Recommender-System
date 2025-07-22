# Semantic-Book-Recommender-System

This project is an intelligent book recommendation system built in Python. Unlike traditional keyword-based search, this engine uses Large Language Models (LLMs) to understand the conceptual meaning of a user's query and matches it with the most relevant books from a dataset of over 7,000 titles.

Key Features
Semantic Vector Search: Leverages the OpenAI API via LangChain to convert book descriptions into high-dimensional embeddings. User queries are vectorized in real-time to find conceptually similar books using a Chroma vector database.

Automated Genre Classification: Utilizes a BART zero-shot classification model from Hugging Face to programmatically standardize over 500 messy, user-generated book categories into a clean, usable set of genres for faceted filtering.

Emotional Tone Analysis: Applies a fine-tuned RoBERTa model to perform sentence-level sentiment analysis on book descriptions. This creates a unique sorting feature, allowing users to find books based on emotional tone (e.g., "suspenseful," "joyful").

Robust Data Pipeline: Features a data processing pipeline built with Pandas for ingesting, cleaning, preprocessing, and feature engineering the book dataset.

Interactive Web UI: The entire system is deployed as a user-friendly web application using Gradio, featuring an intuitive interface with a query input, category filters, and emotional tone sorters.

Tech Stack
Language: Python

Core Libraries: Pandas, NumPy

LLM & NLP Frameworks: LangChain, Hugging Face Transformers

Models: OpenAI text-embedding-ada-002, BART (zero-shot), RoBERTa (fine-tuned)

Vector Database: ChromaDB

Web Dashboard: Gradio

How It Works
Data Ingestion & Cleaning: The initial dataset is loaded, cleaned, and preprocessed using a Pandas pipeline.

Embedding & Indexing: Each book description is passed through an OpenAI embedding model. The resulting vectors are stored and indexed in a Chroma vector database.

Query Processing: A user's natural language query is vectorized using the same embedding model.

Semantic Search & Filtering: The query vector is used to perform a similarity search in ChromaDB. The results are then filtered by genre (classified by BART) and sorted by emotional tone (analyzed by RoBERTa).

Presentation: The final recommendations are displayed in an interactive Gradio dashboard.
