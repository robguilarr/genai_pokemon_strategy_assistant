from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.pydantic_v1 import BaseModel, Field


class IntentTagger(BaseModel):
    """Tag the piece of text with particular intent type, and detect the text
    structure describing the intent"""

    intent_type: str = Field(
        description="Intent type expressed in the text, must take one of the "
        "functionalities as: 'defense_suggestion', 'information_request' or "
        "'squad_build'",
        default=None,
    )
    intent_structure: str = Field(
        description="The type of sentence structure used to detect the intent must "
        "take one of the values: 'pokemon_names', 'natural_language_question' or "
        "'natural_language_description'",
        default=None,
    )


intent_parser = PydanticOutputFunctionsParser(pydantic_schema=IntentTagger)
