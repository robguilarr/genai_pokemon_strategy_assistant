from typing import Dict, Any, List
import re
from textwrap import dedent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from parsers.info_output_parser import PokemonEntity

from setup_loader import SetupLoader

app_setup = SetupLoader()
base_llm, logger, global_conf, prompt_template_library = (
    app_setup.chat_openai,
    app_setup.logger,
    app_setup.global_conf,
    app_setup.prompt_template_library,
)


def format_docs(docs: Any) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _get_retrieval_qa_chain(qa_prompt: str) -> RunnableParallel:
    """Create a chain that can be used to performa semantic queries over FAISS Vector
    Store.
    Args:
        qa_prompt (str): QA prompt to be used.
    Returns:
        RunnableParallel: Language model chain structured as RunnableParallel.
    """
    new_vectorstore = FAISS.load_local(
        folder_path=f"{global_conf['VECTOR_STORE_PATH']}/pokedex_index_react",
        embeddings=OpenAIEmbeddings(),
    )
    retriever = new_vectorstore.as_retriever()
    prompt = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=[
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["context", "question"],
                    template=dedent(prompt_template_library[qa_prompt]),
                )
            )
        ],
    )

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | base_llm
        | StrOutputParser()
    )
    # Adding sources to return
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain)

    return rag_chain_with_source


def retrieval_qa_agent(
    user_query: str = None,
    qa_prompt: str = None,
    pokemon_list: List[PokemonEntity] = None,
) -> Dict[str, Any]:
    """ -- RAG Generation technique --
    RetrievalQA chain handler to performa semantic queries over FAISS Vector Store.
    Args:
        user_query (str, optional): User query to be used. Defaults to None.
        qa_prompt (str, optional): QA prompt to be used. Defaults to None.
        pokemon_list (List[PokemonEntity], optional): List of Pokémon entities.
        Defaults to None.
    Returns:
        Dict[str, Any]: Dictionary containing the Pokémon entity and its description,
        or the relevant answer for the given question.
    Note:
        More information at:
        https://python.langchain.com/docs/use_cases/question_answering/quickstart
    """
    outputs = {}
    pokemon_names = [str(pokemon.name) for pokemon in pokemon_list]

    rag_chain_with_source = _get_retrieval_qa_chain(qa_prompt)

    for pokemon in pokemon_names:
        outputs[pokemon] = rag_chain_with_source.invoke(
            dedent(prompt_template_library[user_query]).format(pokemon_name=pokemon)
        )

    return outputs


def clean_string(s: str) -> str:
    """Clean a string by removing non-alphabetic characters and trailing spaces.
    Args:
        s (str): String to be cleaned.
    Returns:
        str: Cleaned string.
    """
    cleaned = re.sub(r"[^a-zA-Z\s]", "", s)
    cleaned = cleaned.strip()
    return cleaned


def defensive_qa_agent(
    user_query: str = None,
    qa_prompt: str = None,
    pokemon_info: Dict[str, Any] = None,
    pokemon_list: List[PokemonEntity] = None,
) -> Dict[str, Any]:
    """ -- RAG Generation technique --
    DefensiveQA chain handler to performa semantic queries over FAISS Vector Store.
    Args:
        user_query (str, optional): User query to be used. Defaults to None.
        qa_prompt (str, optional): QA prompt to be used. Defaults to None.
        pokemon_info (Dict[str, Any], optional): Dictionary containing the Pokémon
        entity and its description. Defaults to None.
        pokemon_list (List[PokemonEntity], optional): List of Pokémon entities.
        Defaults to None.
    Returns:
        Dict[str, Any]: Dictionary containing the Pokémon entity and its description,
        or the relevant answer for the given question.
    Note:
        More information at:
        https://python.langchain.com/docs/use_cases/question_answering/quickstart
    """
    outputs = {}

    pokemon_list = [pokemon.name for pokemon in pokemon_list]
    damage_relations = {
        pokemon: pokemon_info[pokemon]["damage_relations"] for pokemon in pokemon_list
    }
    user_query = dedent(prompt_template_library[user_query])

    rag_chain_with_source = _get_retrieval_qa_chain(qa_prompt)

    pokemon_retrieved = []
    for pokemon in pokemon_list:
        query = _format_damage_relation_query(
            damage_relations=damage_relations,
            pokemon=pokemon,
            user_query=user_query,
            pokemon_retrieved=pokemon_retrieved,
        )
        outputs[pokemon] = rag_chain_with_source.invoke(dedent(query))
        outputs[pokemon]["answer"] = clean_string(outputs[pokemon]["answer"])

        if outputs[pokemon]["answer"] == "None":
            query = _format_damage_relation_query(
                damage_relations=damage_relations,
                pokemon=pokemon,
                user_query=user_query,
                pokemon_retrieved=pokemon_retrieved,
                empty_answer=True,
            )
            outputs[pokemon] = rag_chain_with_source.invoke(dedent(query))
            outputs[pokemon]["answer"] = clean_string(outputs[pokemon]["answer"])

        pokemon_retrieved.append(outputs[pokemon]["answer"])

    return outputs


def _format_damage_relation_query(
    damage_relations: Dict,
    pokemon: str,
    user_query: str,
    pokemon_retrieved: List[str],
    empty_answer: bool = False,
) -> str:
    """Format the user query to request the damage relations of a Pokémon.
    Args:
        damage_relations (Dict): Dictionary containing the damage relations of a Pokémon.
        pokemon (str): Pokémon name.
        user_query (str): User query to be formatted.
        pokemon_retrieved (List[str]): List of Pokémon names that have been retrieved.
        empty_answer (bool, optional): Flag to indicate if the answer is empty.
        Defaults to False.
    Returns:
        str: Formatted user query.
    """
    if pokemon_retrieved:  # Validation to avoid adding the same names twice
        user_query += f"\nThe Pokémon must NOT be {', or '.join(pokemon_retrieved)}."

    damage_relations = damage_relations[pokemon]

    relation_messages = {  # Order sensitive
        "double_damage_from": "\nCould be a {double_damage_from} type.",
        "double_damage_to": "\nMust NOT be a {double_damage_to} type.",
        "half_damage_to": "\nCould be a {half_damage_to} type.",
        "half_damage_from": "\nMust NOT be a {half_damage_from} type.",
        "no_damage_to": "\nCould be a {no_damage_to} type.",
        "no_damage_from": "\nMust NOT be a {no_damage_from} type.",
    }

    seen_values = set()  # Filter out duplicated 'types'
    filtered_damage_relations = {
        key: value
        for key, value in damage_relations.items()
        if value not in seen_values and (seen_values.add(value) or True)
    }

    if not empty_answer:  # When the answer is not empty, request all details
        for relation_type, message in relation_messages.items():
            if relation_type in filtered_damage_relations.keys():
                user_query += message

        user_query += prompt_template_library["stage_4_instructions"]
        return user_query.format(
            pokemon_name=pokemon.upper(), **filtered_damage_relations
        )

    else:  # When the answer is empty, just request for the relevant Pokémon types
        priority_keys = ["double_damage_from", "no_damage_to"]
        filtered_damage_relations = {
            k: damage_relations[k] for k in priority_keys if k in damage_relations
        }

        for key in filtered_damage_relations:
            user_query += relation_messages[key]

        user_query += prompt_template_library["stage_4_instructions"]
        return user_query.format(
            pokemon_name=pokemon.upper(), **filtered_damage_relations
        )
