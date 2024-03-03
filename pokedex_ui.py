import streamlit as st
from conf.config_loader import default_messages
from typing import List, Dict
from textwrap import dedent
import logging
import os
import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a sidebar with a title and some text
with st.sidebar:
    display_json = st.radio("JSON Mode", [False, True], index=0)
    st.markdown(dedent(default_messages["function_desc"]))
    "[Github Repository](https://github.com/robguilarr)"
    "[![View the source code](https://img.shields.io/github/followers/robguilarr)](https://github.com/robguilarr)"


col1, col2 = st.columns(spec=[0.1, 0.9])
with col1:
    st.image("assets/static/pokemon_go.png", width=70)
with col2:
    st.title("GenAI-Driven Pokémon Index")

# Create a chat container
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello there, how can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


def display_content(
    body_list: List[str], sprites_dict: Dict[str, List[str]], nlp_answer=False
):
    """Display the response content.
    Args:
        body_list (List[str]): List of responses as text.
        sprites_dict (Dict[str, List[str]]): Dictionary of lists of components.
        nlp_answer (bool): NLP answer flag.
    """
    with st.container():
        for i in range(len(body_list)):
            st.write(body_list[i])

            if not nlp_answer:
                st.session_state.messages.append(
                    {"role": "assistant", "content": body_list[i]}
                )

            if sprites_dict:
                current_sprites = list(sprites_dict.values())[i]
                cols = st.columns(len(current_sprites))
                for col, sprite in zip(cols, current_sprites):
                    col.image(sprite, use_column_width=True)


if user_query := st.chat_input():

    logger.info(f"Input query: {user_query}")

    # Append the user query to the chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    logger.info("Executing the intent handler with LLM model")
    try:
        BASE_URL = f"{os.environ['API_URL']}:{os.environ['API_PORT']}"
        api_response = requests.post(
            f"{BASE_URL}/intent_query/", json={"user_query": user_query}
        )
        response = api_response.json()["response"]
    except Exception as e:
        st.error(f"Error on API call: {e}")
        st.stop()

    # Display the response
    header = response.get("header", "")
    body = response.get("body", None)
    sprites = response.get("sprites", {})
    sections = response.get("sections", None)
    intent_type = response.get("intent_type", "")
    intent_structure = response.get("intent_structure", "")
    no_intent = response.get("no_intent", False)
    error = response.get("error", False)

    if display_json:
        st.json(response, expanded=False)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        if no_intent:
            st.chat_message("assistant").write(header)
            st.session_state.messages.append({"role": "assistant", "content": header})
        elif error:
            st.chat_message("assistant").write(header)
            st.session_state.messages.append({"role": "assistant", "content": header})
        else:
            st.chat_message("assistant").write(header)
            if intent_type == "information_request":
                if intent_structure in [
                    "natural_language_question",
                    "natural_language_description",
                ]:
                    display_content(body, sprites, nlp_answer=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": header}
                    )
                elif intent_structure == "pokemon_names":
                    display_content(body, sprites)

            elif intent_type in ["defense_suggestion", "squad_build"]:
                display_content(body, sprites)
