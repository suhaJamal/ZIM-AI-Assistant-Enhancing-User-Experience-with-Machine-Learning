"""
Author: Suha Islaih
Project: ZIMJS Chatbot AI (Final Project for Machine Learning Certificate)
Date: Sept 11, 2024
Description: This script sets up the Gradio-based user interface for the ZIM chatbot.
The chatbot provides responses based on user queries related to the ZIM JavaScript framework.
Key features include handling user input, managing chat history, downloading chat transcripts,
and providing a clean, customizable UI theme. The code integrates a pre-trained AI model
to generate responses and a custom theme for a tailored UI experience.

This code is part of the ZIMJS Chatbot AI project developed as a final project
for the Machine Learning Certificate at York University.
For educational purposes and demonstration of skills in Machine Learning, NLP,
and RAG (Retrieval-Augmented Generation).

How to edit:
- If you want to modify the model's response, look into the `generate_response` function in the
  imported module `zim_ai_assistant.py`.
- To adjust the chat interface design, change the Gradio components inside the `gr.Blocks`.
- For UI theme customization, modify the `theme` object or update the `custom_css` variable.
- If the chat history download behavior needs to be changed, edit the `download_chat` and
  `make_download_visible` functions.
"""

import gradio as gr
from typing import Set, List, Tuple
''' 1- Uncomment the following line to enable the full ZIM chatbot with conversation history and standalone question generation.'''
#from zim_ai_assistant import generate_response

''' 2- Comment the following line to disable the simplified ZIM chatbot (without conversation history and standalone question generation)'''
from zim_ai_assistant_simple import generate_response
import tempfile

# Helper function for formatting sources
def create_sources_string(source_urls: Set[str]) -> str:
    """
    Creates a formatted string for displaying source URLs.

    Args:
        source_urls (Set[str]): A set of source URLs from which the chatbot gathered information.

    Returns:
        str: A formatted string listing the sources, or an empty string if no sources are provided.

    Notes:
        This function sorts the source URLs and formats them with a numeric index for easier readability.
    """
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"
    return sources_string

# Define the function to handle user input and generate a response
def chat_interface(prompt, chat_answers_history, user_prompt_history, chat_history):
    """
    Manages the chat interaction between the user and the chatbot.

    Args:
        prompt (str): User's input query.
        chat_answers_history (list): The history of all responses generated by the chatbot.
        user_prompt_history (list): The history of all user inputs.
        chat_history (list): A list of tuples containing chat history (human, AI).

    Returns:
        tuple: Updated chat interface data (chat, chat_answers_history, user_prompt_history, chat_history, prompt reset).

    Notes:
        The function calls `generate_response()` to get the AI's reply to the user's query, formats the response,
        and updates the chat history. This function also prepares the chat for display in the Gradio UI.
    """
    # Add user message to chat history
    user_prompt_history.append(prompt)
    chat_history.append(("human", prompt))

    # Generate response
    generated_response, sources = generate_response(
        # 3- Uncomment the following line to enable the full ZIM chatbot with conversation history and standalone question generation.
        #query=prompt, chat_history=chat_history

        # 4- Comment the following line to disable the simplified ZIM chatbot (without conversation history and standalone question generation)
        query=prompt
    )
    formatted_response = f"{generated_response} \n\n {create_sources_string(sources)}"

    # Add assistant response to chat history
    chat_answers_history.append(formatted_response)
    chat_history.append(("ai", generated_response))

    # Return updated chat history to display in Gradio
    chat = []
    for user_query, response in zip(user_prompt_history, chat_answers_history):
        chat.append((user_query, None))
        chat.append((None, response))

    return chat, chat_answers_history, user_prompt_history, chat_history, ""

def clear_chat():
    """
    Clears the chat history and resets all states.

    Returns:
        tuple: Empty lists for chatbox, prompt, chat_answers_history, user_prompt_history, and chat_history,
        and hides the download link.

    Notes:
        This function is triggered when the 'Clear Chat' button is clicked.
    """
    return [], "", [], [], [], gr.update(visible=False, value=None)

def download_chat(chat_answers_history, user_prompt_history):
    """
    Saves the entire chat history into a temporary text file for download.

    Args:
        chat_answers_history (list): The history of chatbot responses.
        user_prompt_history (list): The history of user queries.

    Returns:
        str: Path to the temporary file where chat history is saved.

    Notes:
        This function generates a downloadable text file containing the full conversation.
    """
    full_chat = ""
    for user_query, response in zip(user_prompt_history, chat_answers_history):
        full_chat += f"User: {user_query}\nAI: {response}\n\n"

    # Save the chat to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(full_chat.encode('utf-8'))
        tmp_file_path = tmp_file.name

    return tmp_file_path

def make_download_visible(chat_answers_history, user_prompt_history):
    """
    Makes the chat download link visible after generating the chat transcript.

    Args:
        chat_answers_history (list): The history of chatbot responses.
        user_prompt_history (list): The history of user queries.

    Returns:
        gr.update: Update to make the download link visible with the generated file path.
    """
    file_path = download_chat(chat_answers_history, user_prompt_history)  # Generate the file and return the file path
    return gr.update(value=file_path, visible=True)

# Theme and custom CSS for the Gradio UI
theme = gr.themes.Default(
    primary_hue="yellow",
    secondary_hue="blue",
    neutral_hue="gray",
    font=("Roboto", "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=("Roboto Mono", "ui-monospace", "Consolas", "monospace"),
).set(
    background_fill_primary="#000000",
    background_fill_secondary="#111111",
    body_text_color="#FFFFFF",
    block_title_text_color="#FFFFFF",
    block_label_text_color="#FFFFFF",
    input_background_fill="#111111",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="*neutral_950",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_200",
    button_secondary_text_color="*neutral_800",
    block_shadow="*shadow_drop_lg",
    checkbox_background_color="*neutral_800",
    checkbox_background_color_selected="*primary_600",
    checkbox_border_color="*neutral_700",
    checkbox_border_color_focus="*primary_500",
    checkbox_border_color_selected="*primary_600",
    # Custom settings using theme builder variables
    code_background_fill="#222222",          # Background color for code block wrapper
)
custom_css = """
.flex-wrap.user.svelte-1e1jlin.svelte-1e1jlin.svelte-1e1jlin {
    background-color: transparent !important;
}

.code_wrap pre {
  background-color: transparent;
  border: 2px solid white; 
  border-radius: 5px; 
  }
"""

# Gradio UI setup
with gr.Blocks(theme=theme, css=custom_css) as demo:
    gr.Markdown(
        "<h1 style='text-align: center; color: #fff;'>ZIM Chatbot</h1> <h4 style='text-align: center'> Ask me anything about the ZIM JavaScript Canvas Framework</h4>")
    chatbox = gr.Chatbot(height=600, avatar_images=(None, None), elem_id="component-02")
    prompt = gr.Textbox(placeholder="Enter your prompt here..", label="")

    # Initialize session-specific states
    chat_answers_history = gr.State([])
    user_prompt_history = gr.State([])
    chat_history = gr.State([])

    # Create buttons in a single row
    with gr.Row():
        submit_button = gr.Button("Send")
        clear_button = gr.Button("Clear Chat")
        generate_link_button = gr.Button("Download Chat History")

    download_file = gr.File(label="Download Chat History", visible=False)

    # Link buttons to their functions
    submit_button.click(chat_interface, inputs=[prompt, chat_answers_history, user_prompt_history, chat_history],
                        outputs=[chatbox, chat_answers_history, user_prompt_history, chat_history, prompt])
    prompt.submit(chat_interface, inputs=[prompt, chat_answers_history, user_prompt_history, chat_history],
                  outputs=[chatbox, chat_answers_history, user_prompt_history, chat_history, prompt])
    clear_button.click(clear_chat, outputs=[chatbox, prompt, chat_answers_history, user_prompt_history, chat_history,
                                            download_file])
    generate_link_button.click(make_download_visible, inputs=[chat_answers_history, user_prompt_history],
                               outputs=download_file)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
