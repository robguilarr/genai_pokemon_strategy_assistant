from dataclasses import dataclass, field
from langchain_core.runnables import RunnableSequence
from parsers.info_output_parser import PokemonEntity, PokemonEntityList
from agents.information_retrieval_agent import api_retrieval_agent
from agents.rag_qa_agent import (
    retrieval_qa_agent,
    defensive_qa_agent,
    _get_retrieval_qa_chain,
)
from src.common.response_template import ResponseTemplate
from agents.pydantic_agent import (
    get_intent_chain,
    get_pokemon_entity_chain,
    get_no_intent_chain,
)
from setup_loader import SetupLoader

app_setup = SetupLoader()
base_llm, logger, global_conf, prompt_template_library = (
    app_setup.chat_openai,
    app_setup.logger,
    app_setup.global_conf,
    app_setup.prompt_template_library,
)


@dataclass
class IntentHandler:
    """Class to handle the intent of the user input and route it to the corresponding
    agent.
    Attributes:
        pokemon_entity_chain (RunnableSequence, optional): Chain to gather Pokémon
        entities. Defaults to get_pokemon_entity_chain.
        intent_chain (RunnableSequence, optional): Chain to tag the intent type and
        structure. Defaults to get_intent_chain.
        no_intent_chain (RunnableSequence, optional): Chain to handle the case where no
        intent is found. Defaults to get_no_intent_chain.
        user_input (str, optional): User input. Defaults to "".
    Other agents (non-attributes):
        QA Agents (RunnableParallel): used to collect information and suggestions
        based on a method, however internally a RunnableParallel is used to perform
        tasks in parallel (context retrieval).
        API Agents (RunnableParallel): Used to collect information from an API using
        a tool selection method using RunnableSequence.
    """

    response_template: ResponseTemplate = field(default_factory=ResponseTemplate)
    pokemon_entity_chain: RunnableSequence = field(
        default_factory=get_pokemon_entity_chain
    )
    intent_chain: RunnableSequence = field(default_factory=get_intent_chain)
    no_intent_chain: RunnableSequence = field(default_factory=get_no_intent_chain)
    user_input: str = field(default_factory=str)

    def run(self) -> ResponseTemplate:
        logger.info("Stage 0: `Tagging` intent type and structure")
        intent_chain = self.intent_chain
        try:
            intent_chain_output = intent_chain.invoke({"input": self.user_input})
            assert intent_chain_output.intent_type is not None, "No intent found"
        except AssertionError as e:
            logger.error(f"Error: {e}")
            self.response_template.error = True
            return self.response_template

        self.response_template.intent_type = intent_chain_output.intent_type
        self.response_template.intent_structure = intent_chain_output.intent_structure

        if intent_chain_output.intent_type == "information_request":
            logger.info("Branch 1: Routing `information request` intent")
            return self.handle_information_intent(
                structure=intent_chain_output.intent_structure
            )

        elif intent_chain_output.intent_type == "defense_suggestion":
            logger.info("Branch 2: Routing `defense suggestion` intent")
            return self.handle_defense_intent()

        elif intent_chain_output.intent_type == "squad_build":
            logger.info("Branch 3: Routing `squad builder` intent")
            return self.handle_squad_build_intent()

        else:
            logger.error(f"No intent found: {intent_chain_output.intent_type}")
            self.response_template.no_intent = True
            return self.handle_no_intent()

    def handle_information_intent(self, structure: str) -> ResponseTemplate:
        """Handle the `information` intent and route it to the corresponding agent.
        Args:
            structure (str): Intent textual structure.
        Returns:
            ResponseTemplate: Response template.
        """
        if structure == "pokemon_names":
            try:
                logger.info("Sub Branch 1.1: Routing `pokemon name` structure")
                # 1.1.1. Gather Pokémon entity
                pokemon_entity_chain = self.pokemon_entity_chain
                pokemon_entities_output = pokemon_entity_chain.invoke(
                    {"input": self.user_input}
                )
                assert pokemon_entities_output.name_list, "No Pokémon entity found"
                # 1.1.2. Append API info
                pokemon_info = api_retrieval_agent(
                    pokemon_entity_list=pokemon_entities_output,
                    prompt="stage_2_information_api_search_template",
                )
                # 1.1.3. Append Pokémon Description (Semantic Search)
                pokemon_descriptions = retrieval_qa_agent(
                    user_query="stage_3_query_template",
                    qa_prompt="stage_3_retrieval_qa_template",
                    pokemon_list=pokemon_entities_output.name_list,
                )
                self.response_template.pokemon_info = pokemon_info
                self.response_template.pokemon_descriptions = pokemon_descriptions
            except AssertionError as e:
                logger.error(f"Error: {e}")
                self.response_template.error = True

            return self.response_template

        elif structure == "natural_language_question":
            try:
                logger.info(
                    "Sub Branch 1.2: Routing `natural language question` structure"
                )
                # 1.2.1. Gather direct answer from QA
                qa_chain = _get_retrieval_qa_chain(
                    qa_prompt="stage_3_retrieval_qa_template"
                )
                answer = qa_chain.invoke(self.user_input)
                # 1.2.2. Gather Pokémon entity
                pokemon_entity_chain = self.pokemon_entity_chain
                pokemon_entities_output = pokemon_entity_chain.invoke(
                    {"input": self.user_input}
                )
                assert pokemon_entities_output.name_list, "No Pokémon entity found"
                # 1.2.3. Append API info
                pokemon_info = api_retrieval_agent(
                    pokemon_entity_list=pokemon_entities_output,
                    prompt="stage_2_information_api_search_template",
                )
                self.response_template.nlp_answer = answer
                self.response_template.pokemon_info = pokemon_info
            except AssertionError as e:
                logger.error(f"Error: {e}")
                self.response_template.error = True

            return self.response_template

        elif structure == "natural_language_description":
            # NOTE: This section requires wiki with descriptions
            try:
                logger.info(
                    "Sub Branch 1.3: Routing `natural language description` structure"
                )
                # 1.3.1. Gather direct answer from QA
                qa_chain = _get_retrieval_qa_chain(
                    qa_prompt="stage_3_retrieval_qa_template"
                )
                answer = qa_chain.invoke(self.user_input)
                assert answer["answer"] != "None", "No answer found"
                # 1.3.2. Gather Pokémon entity
                pokemon_entity_chain = self.pokemon_entity_chain
                pokemon_entities_output = pokemon_entity_chain.invoke({"input": answer})
                assert pokemon_entities_output.name_list, "No Pokémon entity found"
                # 1.3.3. Append API info
                pokemon_info = api_retrieval_agent(
                    pokemon_entity_list=pokemon_entities_output,
                    prompt="stage_2_information_api_search_template",
                )
                self.response_template.nlp_answer = answer
                self.response_template.pokemon_info = pokemon_info
            except AssertionError as e:
                logger.error(f"Error: {e}")
                self.response_template.error = True

            return self.response_template

        else:
            logger.info("No intent structure found")
            self.response_template.no_intent = True

            return self.response_template

    def handle_defense_intent(self) -> ResponseTemplate:
        """Handle the `defense intent` and route it to the corresponding agent.
        Returns:
            ResponseTemplate: Response template.
        """
        try:
            # 2.1. Gather Pokémon entity
            opponent_pokemon_entity_chain = self.pokemon_entity_chain
            opponent_pokemon_entities_output = opponent_pokemon_entity_chain.invoke(
                {"input": self.user_input}
            )
            assert opponent_pokemon_entities_output.name_list, "No Pokémon entity found"

            # 2.2. Append API info
            opponent_pokemon_info = api_retrieval_agent(
                pokemon_entity_list=opponent_pokemon_entities_output,
                prompt="stage_2_information_api_search_template",
            )
            # 2.3. Append Pokémon Defense Suggestion (Semantic Search)
            pokemon_defense_suggestion = defensive_qa_agent(
                user_query="stage_4_defensive_recommendation_template",
                qa_prompt="stage_3_retrieval_qa_template",
                pokemon_info=opponent_pokemon_info,
                pokemon_list=opponent_pokemon_entities_output.name_list,
            )
            # 2.4. Gather Pokémon entity from API
            pokemon_defense_list = [
                PokemonEntity(name=value["answer"])
                for key, value in pokemon_defense_suggestion.items()
                if value["answer"] != "None"
            ]
            pokemon_defense_info = api_retrieval_agent(
                pokemon_entity_list=PokemonEntityList(name_list=pokemon_defense_list),
                prompt="stage_2_information_api_search_template",
            )
            self.response_template.pokemon_defense_info = pokemon_defense_info
        except AssertionError as e:
            logger.error(f"Error: {e}")
            self.response_template.error = True

        return self.response_template

    def handle_squad_build_intent(self) -> ResponseTemplate:
        """Handle the `squad build` intent and route it to the corresponding agent.
        Returns:
            ResponseTemplate: Response template.
        """
        try:
            # 3.1. Gather Pokémon entity
            opponent_pokemon_entity_chain = self.pokemon_entity_chain
            opponent_pokemon_entities_output = opponent_pokemon_entity_chain.invoke(
                {"input": self.user_input}
            )
            assert opponent_pokemon_entities_output.name_list, "No Pokémon entity found"
            # 3.2. Append API info
            opponent_pokemon_info = api_retrieval_agent(
                pokemon_entity_list=opponent_pokemon_entities_output,
                prompt="stage_2_information_api_search_template",
            )
            # 3.3. Append Pokémon Defense Suggestion (Semantic Search)
            pokemon_defense_suggestion = defensive_qa_agent(
                user_query="stage_4_defensive_recommendation_template",
                qa_prompt="stage_3_retrieval_qa_template",
                pokemon_info=opponent_pokemon_info,
                pokemon_list=opponent_pokemon_entities_output.name_list,
            )
            # 3.4. Gather Pokémon entity from API
            pokemon_squad_list = [
                PokemonEntity(name=value["answer"])
                for key, value in pokemon_defense_suggestion.items()
                if value["answer"] != "None"
            ]
            pokemon_squad_info = api_retrieval_agent(
                pokemon_entity_list=PokemonEntityList(name_list=pokemon_squad_list),
                prompt="stage_2_information_api_search_template",
            )
            self.response_template.pokemon_squad_info = pokemon_squad_info
        except AssertionError as e:
            logger.error(f"Error: {e}")
            self.response_template.error = True

        return self.response_template

    def handle_no_intent(self) -> ResponseTemplate:
        """Handle the `no intent` and route it to the corresponding default GPT agent.
        Returns:
            ResponseTemplate: Response template.
        """
        try:
            no_intent_chain = self.no_intent_chain
            intent_chain_output = no_intent_chain.invoke({"input": self.user_input})
            self.response_template.nlp_answer = intent_chain_output
        except AssertionError as e:
            logger.error(f"Error: {e}")
            self.response_template.error = True

        return self.response_template
