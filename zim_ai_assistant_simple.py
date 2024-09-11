import openai
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from typing import Any, List
import pickle
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
openai.api_key = os.getenv("OPENAI_API_KEY")
data = pd.read_csv('datasets/cleaned_zimjs_with_messages.csv')
data = data.sample(frac=1).reset_index(drop=True)

embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

file_path = "vector-DB/"  #change dir to your project folder

# Load FAISS index and prompt template
db = FAISS.load_local(
    file_path + "faiss_index",
    HuggingFaceEmbeddings(model_name=embedding_model_name),
    allow_dangerous_deserialization=True
)

with open(file_path + "prompt_template.pkl", "rb") as f:
    prompt_template = pickle.load(f)

# Initialize the retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})

# Define a function to interact with GPT-3.5-turbo
def gpt3_5_turbo_generate(context, question):
    formatted_prompt = prompt_template.format(context=context,
                                              question=question)  # Make sure this line correctly formats the prompt

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
    )

    return response.choices[0].message['content']

# Define the LLM chain using the gpt3_5_turbo_generate function
def gpt3_5_turbo_chain(context, question):
    context = context.replace('\n', ' ').replace('\r', '').strip()
    question = question.replace('\n', ' ').replace('\r', '').strip()
    return gpt3_5_turbo_generate(context, question)

# Function to get the best context using FAISS retriever
def get_combined_context_and_urls(query):
    # Initialize variables to hold combined context and URLs
    combined_context = ""
    urls = []

    # Retrieve documents based on the query
    retrieved_docs = retriever.invoke(query)

    if retrieved_docs:
        # Combine the content of the top `k` documents and save their URLs
        for doc in retrieved_docs[:3]:
            combined_context += doc.page_content + "\n"
            urls.append(doc.metadata.get('source', ''))
    else:
        # If no document is retrieved, fall back to an empty context
        combined_context = ""
        urls = []

    return combined_context, urls

# Function to generate response using GPT-3.5-turbo with the RAG approach
def generate_response(query):
    context, urls = get_combined_context_and_urls(query)

    # Generate the answer using the combined context and the original query
    answer = gpt3_5_turbo_chain(context, query)
    print("=====answer=====")
    print(answer)
    print("====query======")
    print(query)
    return answer, urls

