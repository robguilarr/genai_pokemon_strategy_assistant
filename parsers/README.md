# Parsers Package

This package contains several modules that are responsible for parsing different 
types of output. Here's a brief overview of each module:

## Info Output Parser

The `info_output_parser.py` module is responsible for parsing information about 
Pokémon entities. It uses the PokemonEntity and PokemonEntityList classes to extract 
the names of Pokémon mentioned in the text. More details can be found in the 
function docstrings within the module.

## Intent Output Parser

The `intent_output_parser.py` module is responsible for tagging pieces of text with 
particular intent types and detecting the text structure describing the intent. It 
uses the `IntentTagger` class for this purpose. More details can be found in the 
function docstrings within the module.

## Tooling Output Parser

The `tooling_output_parser.py` module is responsible for parsing tooling entries. It 
uses the `ToolingEntry` class to list Pokémon names mentioned in the input text. 
More details can be found in the function docstrings within the module.
