"""
Author: Suha Islaih
Project: ZIMJS Chatbot AI (Final Project for Machine Learning Certificate)
Date: Sept 11, 2024
Description: This script handles the integration of the FAISS vector store and GPT-3.5-turbo for the ZIMJS chatbot AI.
It retrieves relevant content using FAISS for better context, formulates standalone questions, and generates responses
with GPT-3.5-turbo. The script also manages the interaction between the user and the model, including handling chat history
and ensuring proper response formatting. Key components include FAISS-based document retrieval, prompt formatting, and
response generation using RAG (Retrieval-Augmented Generation).

This code is part of the ZIMJS Chatbot AI project developed as a final project
for the Machine Learning Certificate at York University.
For educational purposes and demonstration of skills in Machine Learning, NLP,
and RAG (Retrieval-Augmented Generation).

How to edit:
- To modify the model's behavior, such as the way questions are transformed into standalone questions,
  adjust the `generate_standalone_question` function.
- If you want to customize the retrieval mechanism, look at the FAISS loading logic and the `retriever`
  initialization.
- The response generation pipeline using GPT-3.5-turbo can be modified in the `gpt3_5_turbo_generate`
  function or `gpt3_5_turbo_chain` function.
- To integrate a different embedding model, change the `embedding_model_name` parameter and ensure the
  model is compatible with FAISS.
"""

import openai
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from typing import Any, Dict, List
import pickle
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the dataset and shuffle it
data = pd.read_csv('datasets/cleaned_zimjs_with_messages.csv')
data = data.sample(frac=1).reset_index(drop=True)

# Define the embedding model to be used with FAISS
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# Path to FAISS index and prompt template files
file_path = "vector-DB/"

# Load the FAISS index and prompt template from saved files
db = FAISS.load_local(
    file_path + "faiss_index",
    HuggingFaceEmbeddings(model_name=embedding_model_name),
    allow_dangerous_deserialization=True
)

# To customize the chatbot's response template, modify the `prompt_template` to suit specific needs.
prompt_template = """
### [INST]
Instruction: You are an expert on the ZIM JS framework. Using the provided context, generate a clear, concise, 
and professional response to the question. Your answer should be easy to understand while maintaining technical 
accuracy. Always use the ZIM JS context provided.

If the user explicitly asks for "full code," "complete code," or mentions "using ZIM template," incorporate 
your answer within the following code template. Otherwise, provide the answer without the template.

ZIM Template:
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>ZIM - Code Creativity</title>
<!-- zimjs.com - JavaScript Canvas Framework -->
<script type="module">
import zim from "https://zimjs.org/cdn/016/zim";
new Frame(FIT, 1024, 768, light, dark, ready);
function ready() {{
    // put code here
}} // end ready
</script>
<meta name="viewport" content="width=device-width, user-scalable=no" />
</head>
<body></body>
</html>

Context:
{context}

### QUESTION:
{question}

[/INST]
"""

# Initialize the PromptTemplate object with the context and question variables
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Initialize the retriever using FAISS
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})


# Function to generate a response using GPT-3.5-turbo
def gpt3_5_turbo_generate(context, question):
    """
    Generates a response from GPT-3.5-turbo based on the provided context and question.

    Args:
        context (str): Combined retrieved context to pass to the model.
        question (str): User's input question, formatted for the model.

    Returns:
        str: The generated response from GPT-3.5-turbo.

    Notes:
        This function sends the formatted prompt to OpenAI's GPT-3.5-turbo and retrieves the response.
    """
    formatted_prompt = prompt_template.format(context=context, question=question)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0,
    )

    return response.choices[0].message['content']


# Function to manage the LLM chain with GPT-3.5-turbo
def gpt3_5_turbo_chain(context, question):
    """
    Formats the input context and question and then generates a response using GPT-3.5-turbo.

    Args:
        context (str): Combined context retrieved from FAISS.
        question (str): Standalone question extracted from the user's query.

    Returns:
        str: The final answer generated by GPT-3.5-turbo.

    Notes:
        This function strips unnecessary whitespace and passes the cleaned context and question
        to the response generation function.
    """
    context = context.replace('\n', ' ').replace('\r', '').strip()
    question = question.replace('\n', ' ').replace('\r', '').strip()
    return gpt3_5_turbo_generate(context, question)


# Function to generate a standalone question using GPT-3.5-turbo
def generate_standalone_question(full_query):
    """
    Converts a full query, which may include chat history and pronouns, into a standalone question.

    Args:
        full_query (str): The user's full query, including current question and chat history.

    Returns:
        str: A cleaned and standalone question without pronouns.

    Notes:
        This function sends the full query to GPT-3.5-turbo, asking the model to replace pronouns
        with the appropriate nouns for clarity.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": '''Please convert the following query, which includes 
            the current question and previous chat history, into a standalone question. Replace any 
            pronouns like 'it' or 'this' with the specific noun they refer to.'''},
            {"role": "user", "content": full_query}
        ]
    )
    return response.choices[0].message['content'].strip()


# Function to retrieve context using FAISS and generate a response
def get_combined_context_and_urls(query, chat_history: List[Dict[str, Any]] = []):
    """
    Retrieves the most relevant context from FAISS and combines it with user chat history.

    Args:
        query (str): The user's current question.
        chat_history (list): A list of chat history containing previous user and AI messages.

    Returns:
        tuple: A tuple containing the combined context, a list of URLs, and the standalone question.

    Notes:
        This function first converts the user's query into a standalone question, then uses the FAISS
        retriever to fetch relevant documents and URLs to provide context for GPT-3.5-turbo.
    """
    formatted_chat_history = []
    for msg in chat_history:
        if isinstance(msg, tuple) and len(msg) == 2:
            role, content = msg
            if role == 'human':
                formatted_chat_history.append({'role': 'user', 'content': content})
            elif role == 'ai':
                formatted_chat_history.append({'role': 'assistant', 'content': content})
        elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            formatted_chat_history.append(msg)
        else:
            raise ValueError(
                "Each item in chat_history should be a tuple with 2 elements or a dictionary with 'role' and 'content' keys")

    full_query = query + " " + " ".join([msg['content'] for msg in formatted_chat_history if msg['role'] == 'user'])

    # Convert the full query into a standalone question
    standalone_question = generate_standalone_question(full_query)

    combined_context = ""
    urls = []

    # Retrieve relevant documents using the standalone question
    retrieved_docs = retriever.invoke(standalone_question)

    if retrieved_docs:
        for doc in retrieved_docs[:3]:
            combined_context += doc.page_content + "\n"
            urls.append(doc.metadata.get('source', ''))
    else:
        combined_context = ""

    return combined_context, urls, standalone_question


# Function to generate a final response with context using GPT-3.5-turbo
def generate_response(query, chat_history: List[Dict[str, Any]] = []):
    """
    Generates a response using RAG (Retrieval-Augmented Generation) by combining FAISS-retrieved
    context and GPT-3.5-turbo response generation.

    Args:
        query (str): User's current question.
        chat_history (list): Previous messages exchanged between the user and the AI.

    Returns:
        tuple: The answer generated by GPT-3.5-turbo and a list of source URLs.

    Notes:
        This function retrieves relevant documents, formats the question, and uses GPT-3.5-turbo
        to generate the final answer.
    """
    context, urls, standalone_question = get_combined_context_and_urls(query, chat_history)
    answer = gpt3_5_turbo_chain(context, standalone_question)
    return answer, urls

