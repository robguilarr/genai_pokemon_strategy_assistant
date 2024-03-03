import pokepy
from typing import Dict, List
from langchain.tools import tool
from langchain_core.tools import ToolException
from parsers.tooling_output_parser import ToolingEntry
from setup_loader import SetupLoader

app_setup = SetupLoader()
base_llm, logger = app_setup.chat_openai, app_setup.logger


@tool("pokemon_api_wrapper", args_schema=ToolingEntry, return_direct=True)
def pokemon_api_wrapper(name_list: List[str]) -> Dict:
    """Useful for when you need to request information from the Pokémon API,
    considering a single Pokémon Entity or several of them as input."""
    client = pokepy.V2Client()
    pokemon_info_collection = {}

    for pokemon_name in name_list:
        try:
            logger.info(" PokemonAPIWrapper: Information Search ")
            pokemon_data = client.get_pokemon(pokemon_name.lower())
            pokemon = pokemon_data[0]
            info = {
                "id": pokemon.id,
                "stats": {stat.stat.name: stat.base_stat for stat in pokemon.stats},
                "height": pokemon.height,
                "weight": pokemon.weight,
                "types": [type_slot.type.name for type_slot in pokemon.types],
                "abilities": [
                    ability_slot.ability.name for ability_slot in pokemon.abilities
                ],
                "sprites": pokemon.sprites.__dict__,
            }

            logger.info(" PokemonAPIWrapper: Types Search ")
            if info["types"]:
                try:
                    # Extract Damage Relations
                    info["damage_relations"] = [
                        client.get_type(type_slot) for type_slot in info["types"]
                    ]
                    # Overwrite Damage Relations
                    damage_relations = info["damage_relations"][0][0].damage_relations
                    info["damage_relations"] = {
                        damage_type: damage_relations.__dict__.get(damage_type)[0].name
                        for damage_type in damage_relations.__dict__.keys()
                        if damage_type != "_subresource_map"
                        and damage_relations.__dict__.get(damage_type) != []
                    }
                except Exception as e:
                    logger.info(f"No 'damage_relations' were extracted: {e}")
                    info["damage_relations"] = {}
                    pass

            pokemon_info_collection[pokemon_name] = info
            logger.info(
                f" PokemonAPIWrapper: Information of '{pokemon_name}' was extracted "
            )

        except Exception as e:
            raise ToolException(f"Tool Error handling '{pokemon_name}': {e}")

    return pokemon_info_collection
