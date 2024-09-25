# ZIM-AI-Assistant-Enhancing-User-Experience-with-Machine-Learning
ZIM JS Chatbot An AI chatbot built using GPT-3.5 Turbo and Retrieval-Augmented Generation (RAG) to enhance user interactions on the ZIMJS website. It answers coding queries with context-aware responses, utilizing FAISS for retrieval and deployed via Gradio for real-time assistance with coding tutorials and discussions.

Based on the attached project summary, here’s a suggested structure for your **README** file:

---

# ZIM JS Chatbot

## Project Overview
The **ZIM JS Chatbot** is an AI-powered assistant designed to enhance the user experience on the ZIMJS website. The chatbot utilizes GPT-3.5 Turbo and Retrieval-Augmented Generation (RAG) to provide context-aware responses to user queries related to coding tutorials and discussions.

## Key Features
- **Conversational AI**: Supports natural language interactions using large language models (LLM).
- **Context-aware responses**: Incorporates conversation history and standalone question generation.
- **Efficient Retrieval**: Leverages FAISS vector store to retrieve relevant documents and provide accurate responses.
- **Deployment**: Available both locally and on Hugging Face, using Gradio as the user interface.

## How to Run the Chatbot
1. Clone the repository:
   ```bash
   git clone https://github.com/suhaJamal/ZIM-AI-Assistant-Enhancing-User-Experience-with-Machine-Learning.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the chatbot:
   ```bash
   python app_session.py
   ```

## Key Files
- **zim_ai_assistant.py**: The full chatbot with conversation history and standalone question generation.
- **zim_ai_assistant_simple.py**: A simplified version without conversation history.
- **vector_db.py**: Manages the FAISS vector store for document retrieval.

## How to Modify the Chatbot
- **Modify prompt template**: Edit `zim_ai_assistant_simple.py` to adjust the model's response behavior.
- **Switch between public/private chatbot**: Modify settings on Hugging Face to change visibility.

Check out the live ZIM Chatbot that I developed using GPT-3.5 Turbo and Retrieval-Augmented Generation (RAG) on the ZIM JS website
https://zimjs.com/bot/?fbclid=IwY2xjawFhJ89leHRuA2FlbQIxMAABHYYWATV-WPmgpP8zCnCY_aZLIEGE_11Y9J3dL1rfiOhbkXD-Hn1QUkazYw_aem_4_nf-vFDkSk9U-spIJnXUw

## Updating the Dataset
To update the chatbot for a new version of the ZIM JS library:
1. Update the dataset with new documentation.
2. Regenerate the FAISS vector index using `vector_db.py`.

