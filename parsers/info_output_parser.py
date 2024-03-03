from typing import List
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain.pydantic_v1 import BaseModel, Field


class PokemonEntity(BaseModel):
    """Extract the piece of text with the Pokémon name"""

    name: str = Field(description="Name of the Pokémon mentioned in text", default=None)


class PokemonEntityList(BaseModel):
    """Extract list of pieces of text with Pokémon names"""

    name_list: List[PokemonEntity] = Field(
        description="List of names of the Pokémon mentioned in text", default=[]
    )


pokemon_entity_parser = PydanticOutputFunctionsParser(pydantic_schema=PokemonEntityList)
