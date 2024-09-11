"""
Author: Suha Islaih
Project: ZIMJS Chatbot AI (Final Project for Machine Learning Certificate)
Date: Sept 11, 2024
Description: This script handles the creation of a FAISS vector store for use in a ZIMJS chatbot AI.
The code processes a dataset of ZIMJS-related content, embeds the documents using a HuggingFace model,
and stores the resulting vectors in a FAISS index for efficient retrieval.

This code is part of the ZIMJS Chatbot AI project developed as a final project
for the Machine Learning Certificate at York University.
For educational purposes and demonstration of skills in Machine Learning, NLP,
and RAG (Retrieval-Augmented Generation).

How to edit:
- To modify the embedding model used with FAISS, update the `embedding_model_name` parameter.
- If you need to adjust the retrieval mechanism, explore the `retriever` initialization and adjust the `search_kwargs`.
"""

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pandas as pd

# Load the dataset
data = pd.read_csv('../datasets/cleaned_zimjs_with_messages.csv')

# Shuffle the data for randomness
data = data.sample(frac=1).reset_index(drop=True)

# Create a list of Document objects, each representing a row from the dataset
docs = [
    Document(
        page_content=f"Content: {content_row['content']}",
        metadata={'source': content_row['url']}
    )
    for _, content_row in data.iterrows()
]

# Define the embedding model to be used with FAISS
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# Create FAISS vector store from the documents and HuggingFace embeddings
db = FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name=embedding_model_name))

# Initialize the retriever using the FAISS vector store
# The retriever will use similarity search and retrieve the top 3 documents
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})

# Save the FAISS index to the local file system for reuse
db.save_local("faiss_index")
