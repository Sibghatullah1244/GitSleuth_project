Repository Q/A Web App

This project is a simple Retrieval-Augmented Generation (RAG) system built with:

Backend: Flask, LangChain, OpenAI, Hugging Face (Sentence Transformers), FAISS

Frontend: HTML + JavaScript

It allows you to ask natural language questions about any public GitHub repository. The system will:

Clone the repository

Preprocess and embed the content

Store embeddings in a FAISS vector database

Generate a response using a RAG pipeline

📂 Project Structure
├── app.py         # Backend (Flask + RAG pipeline)
├── index.html     # Frontend (HTML + JS)
├── README.md      # Documentation

⚙️ How It Works
Step 1: Start the app

Run the Flask backend:

python app.py


Open index.html in your browser.

Step 2: Index a Repository

Paste a GitHub repo URL in the input field.

Click “Start Indexing”.

The backend will:

Clone the repository

Read source files (e.g., .py, .md, .txt)

Preprocess & clean the text (remove noise, format, normalize)

Split into chunks for efficient embedding

Generate embeddings using Hugging Face Sentence Transformers

Store embeddings in FAISS

Create a session ID

The session ID is filled automatically in the frontend so you don’t need to copy it manually.

Step 3: Ask Questions

Enter your question in natural language.

The backend will:

Convert your question into an embedding (using the same embedding model)

Search in FAISS for the most relevant chunks

Pass the chunks + your question to LangChain + OpenAI model

Generate an answer using RAG

The frontend displays:

✅ The Answer (LLM response)

📖 The Sources (which repo files were used to answer)

🛠️ Technologies Used

Flask → lightweight backend API server

LangChain → for chaining LLM with retrieval (RAG)

OpenAI → for final answer generation

Hugging Face Sentence Transformers → for embeddings

FAISS → for efficient similarity search (vector database)

HTML + JavaScript → for frontend

🚀 Example Workflow

Paste repo: https://github.com/psf/requests

Backend clones & indexes it → embeddings stored in FAISS

Ask: “How does requests handle authentication?”

FAISS retrieves chunks about auth.py

OpenAI model generates an answer like:

Requests supports multiple authentication methods.
Basic authentication can be used by passing the `auth` parameter
as a tuple of (username, password). 
It also provides handlers for Digest and custom authentication.


Sources shown:

auth.py, README.md

🖥️ Frontend Overview

The index.html provides two sections:

Index a Repo

Input field for GitHub repo URL

Button to start indexing

Status box showing progress

Ask a Question

Input field for your question

Auto-filled session ID

Answer + Sources displayed below

✅ Features

Works with any public GitHub repo

Automatic cloning, cleaning, embedding, and storing

Session-based indexing (so you can query multiple repos separately)

Shows both answer and sources for transparency

Minimal but functional frontend (HTML/JS)
