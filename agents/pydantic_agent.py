from langchain_core.prompts import ChatPromptTemplate
from parsers.intent_output_parser import intent_parser, IntentTagger
from langchain.schema.output_parser import StrOutputParser
from textwrap import dedent
from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)
from setup_loader import SetupLoader
from parsers.info_output_parser import PokemonEntityList, pokemon_entity_parser
from langchain_core.runnables import RunnableSequence

app_setup = SetupLoader()
base_llm, logger, global_conf, prompt_template_library = (
    app_setup.chat_openai,
    app_setup.logger,
    app_setup.global_conf,
    app_setup.prompt_template_library,
)


def get_intent_chain() -> RunnableSequence:
    """Create a chain that can be used to identify the intent type and structure of a
    user input by using a Tagging approach.
    Returns:
        RunnableSequence: Language model chain structured as RunnableSequence.
    """
    intent_template = dedent(prompt_template_library["stage_0_intent_template"])

    intent_prompt_template = ChatPromptTemplate.from_messages(
        [("system", intent_template), ("human", "{input}")]
    )

    model = base_llm.bind(
        functions=[convert_pydantic_to_openai_function(IntentTagger)],
        function_call={"name": "IntentTagger"},
    )

    return intent_prompt_template | model | intent_parser


def get_pokemon_entity_chain() -> RunnableSequence:
    """Create a chain that can be used to identify the Pokémon entities of a user input
    by using an Extraction approach.
    Returns:
        RunnableSequence: Language model chain structured as RunnableSequence.
    """
    pokemon_template = dedent(
        prompt_template_library["stage_1_pokemon_entity_template"]
    )

    pokemon_entity_template = ChatPromptTemplate.from_messages(
        [("system", pokemon_template), ("human", "{input}")]
    )

    model = base_llm.bind(
        functions=[convert_pydantic_to_openai_function(PokemonEntityList)],
        function_call={"name": "PokemonEntityList"},
    )

    return pokemon_entity_template | model | pokemon_entity_parser


def get_no_intent_chain() -> RunnableSequence:
    """Create a chain that can be used to handle the case where no intent is
    identified. Respond with generic GPT model knowledge.
    Returns:
        RunnableSequence: Language model chain structured as RunnableSequence.
    """
    no_intent_template = dedent(prompt_template_library["stage_0_no_intent_template"])

    return (
        ChatPromptTemplate.from_template(no_intent_template)
        | base_llm
        | StrOutputParser()
    )
