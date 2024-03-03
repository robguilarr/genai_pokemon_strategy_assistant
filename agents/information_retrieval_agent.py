from typing import List, Dict, Any
from tools.tools import pokemon_api_wrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from parsers.info_output_parser import PokemonEntityList
from parsers.tooling_output_parser import tooling_parser
from textwrap import dedent
from setup_loader import SetupLoader

app_setup = SetupLoader()
base_llm, logger, prompt_template_library = (
    app_setup.chat_openai,
    app_setup.logger,
    app_setup.prompt_template_library,
)


def api_retrieval_agent(
    pokemon_entity_list: PokemonEntityList, prompt: str = None
) -> List[Dict[str, Any]]:
    """Use the API retrieval agent to request information about Pokémon entities.
    Args:
        pokemon_entity_list (PokemonEntityList): List of Pokémon entities.
        prompt (str, optional): Prompt to use. Defaults to None.
    Returns:
        List[Dict[str, Any]]: List of dictionaries with the information of the Pokémon
        entities.
    """
    tooling_template = dedent(prompt_template_library[prompt])

    pokemon_names = [str(pokemon.name) for pokemon in pokemon_entity_list.name_list]
    tools = [pokemon_api_wrapper]  # Default structure to add more tools
    tool_map = {tool.name: tool for tool in tools}
    functions = [convert_to_openai_function(t) for t in tools]

    model = base_llm.bind(functions=functions)
    tooling_prompt_template = ChatPromptTemplate.from_messages(
        [("system", tooling_template), ("human", "{input}")]
    )
    chain = tooling_prompt_template | model | tooling_parser
    tooling_result = chain.invoke({"input": pokemon_names})

    try:
        selected_tool = tool_map[tooling_result.tool]
        result = selected_tool(tooling_result.tool_input)
    except Exception as e:
        result = {}
        logger.warning(f"Tool Error Recovering Output: {e}")

    return result
