from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


class ToolingEntry(BaseModel):
    """List of Pokémon names mentioned in the input text"""

    name_list: List[str] = Field(
        description="List of names of the Pokémon names", default=[]
    )


tooling_parser = OpenAIFunctionsAgentOutputParser()
