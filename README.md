# E-Commerce Customer Support Resolution Engine (RAG)

## Overview
This project demonstrates a production-ready Retrieval-Augmented Generation (RAG) pipeline. It acts as an automated customer support agent that answers user queries by explicitly searching and citing private company policy documents, effectively eliminating AI hallucinations.

## The Problem Solved
Large Language Models (LLMs) suffer from a "Knowledge Cutoff" and tend to hallucinate when asked about proprietary or highly specific company data. If an ungrounded bot gives a customer the wrong refund policy, the company is liable. This architecture grounds the AI's generation process entirely in a private Vector Database.

## Tech Stack
* **Python 3.10+**
* **ChromaDB:** A lightweight, local vector database to store document embeddings.
* **Sentence-Transformers:** To convert our text into mathematical vectors (embeddings).
* **Ollama Cloud:** For the LLM generation phase (`gemma4:31b-cloud` or similar).

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file and add your API key:
   `OLLAMA_API_KEY=your_api_key_here`

## Usage
1. Run the ingestion script to populate the vector database with company policies:
   `python ingest.py`
2. Run the main chat engine to test queries against the knowledge base:
   `python chat.py`