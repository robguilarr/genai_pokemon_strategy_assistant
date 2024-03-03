from dataclasses import dataclass, field
from typing import Any, Dict
from conf.config_loader import default_messages
from setup_loader import SetupLoader
import random
from textwrap import dedent

app_setup = SetupLoader()
base_llm, logger, global_conf, prompt_template_library = (
    app_setup.chat_openai,
    app_setup.logger,
    app_setup.global_conf,
    app_setup.prompt_template_library,
)


@dataclass
class ResponseTemplate:
    """Class to handle response templates for displaying the Pokémon UI and API. The
    template structure is completed according to the intent type and intent
    structure, and its components are:
    - header: Initial response from the chatbot.
    - body: Pokémon information.
    - sprites: Pokémon sprites.
    - sections: Pokémon sections (optional).
    Attributes:
        error (bool): Error flag.
        no_intent (bool): No intent flag.
        intent_type (str): Intent type.
        intent_structure (str): Intent structure.
        nlp_answer (str): NLP answer.
        pokemon_info (Dict[str, Any]): Pokémon information.
        pokemon_defense_info (Dict[str, Any]): Pokémon defense information.
        pokemon_descriptions (Dict[str, Any]): Pokémon descriptions.
        pokemon_squad_info (Dict[str, Any]): Pokémon squad information.
        response (Dict[str, Any]): Response.
    """

    error: bool = field(default=False)
    no_intent: bool = field(default=False)
    intent_type: str = field(default_factory=str)
    intent_structure: str = field(default_factory=str)
    nlp_answer: str = field(default_factory=str)
    pokemon_info: Dict[str, Any] = field(default_factory=dict)
    pokemon_defense_info: Dict[str, Any] = field(default_factory=dict)
    pokemon_descriptions: Dict[str, Any] = field(default_factory=dict)
    pokemon_squad_info: Dict[str, Any] = field(default_factory=dict)
    response: Dict[str, Any] = field(default_factory=dict)

    @property
    def template_structure(self):
        """Property to orchestrate the template structure builders."""
        self.response = dict(
            header=None,
            body=None,
            sprites=None,
            intent_type=self.intent_type,
            intent_structure=self.intent_structure,
        )
        if self.intent_type == "information_request":
            self._header_template()
            return self.build_information_request_template()
        elif self.intent_type == "defense_suggestion":
            self._header_template()
            return self.build_defense_suggestion_template()
        elif self.intent_type == "squad_build":
            self._header_template()
            return self.build_squad_build_template()
        elif self.no_intent:
            return self.build_no_intent_template()
        else:
            return self.build_no_intent_template()

    def build_information_request_template(self) -> Dict[str, Any]:
        """Builder for the information request template.
        Returns:
            Dict[str, Any]: Response template populated.
        """
        if self.error:
            return self.build_error_template()

        if self.intent_structure == "pokemon_names":
            self._pokemon_info_template()
            self._pokemon_descriptions_template()
            return self.response

        elif self.intent_structure == "natural_language_question":
            self._nlp_answer_template()
            self._pokemon_info_template()
            return self.response

        elif self.intent_structure == "natural_language_description":
            self._nlp_answer_template()
            self._pokemon_info_template()
            return self.response

        else:
            return self.response

    def build_defense_suggestion_template(self) -> Dict[str, Any]:
        """Builder for the defense suggestion template.
        Returns:
            Dict[str, Any]: Response template populated.
        """
        if self.error:
            return self.build_error_template()

        self._pokemon_info_template(defense=True)
        return self.response

    def build_squad_build_template(self) -> Dict[str, Any]:
        """Builder for the squad build template.
        Returns:
            Dict[str, Any]: Response template populated.
        """
        if self.error:
            return self.build_error_template()

        self._pokemon_info_template(squad_defense=True)
        return self.response

    def build_error_template(self) -> Dict[str, Any]:
        """Builder for alert message when no template is found.
        Returns:
            Dict[str, Any]: Response template populated.
        """
        self.response["header"] = default_messages["alert_no_answer"]
        return self.response

    def build_no_intent_template(self) -> Dict[str, Any]:
        """Builder for alert message when no template is found. Return answer from
        the LLM model directly.
        Returns:
            Dict[str, Any]: Response template populated.
        """
        logger.warning(f"Unknown intent type: {self.intent_type}")
        if self.error:
            return self.build_error_template()

        self.response["header"] = default_messages["base_llm_answer"] + dedent(
            self.nlp_answer
        )
        return self.response

    def _nlp_answer_template(self):
        """Helper that populates the NLP answer template."""
        self.response["header"] = dedent(self.nlp_answer["answer"]) + "\n"

    def _pokemon_descriptions_template(self):
        """Helper that populates the Pokémon descriptions template."""
        listed_responses = list(self.pokemon_descriptions.values())

        if not self.response["body"]:
            self.response["body"] = ["" for _ in range(len(listed_responses))]

        for i in range(len(self.pokemon_descriptions)):
            self.response["body"][i] += "\n" + dedent(listed_responses[i]["answer"])

    def _header_template(self):
        """Helper that populates the header template."""
        self.response["header"] = dedent(
            random.choice(list(default_messages["initial_responses"].values()))
        )

    def _pokemon_info_template(
        self, defense: bool = False, squad_defense: bool = False
    ):
        """Helper that populates the Pokémon information template.
        Args:
            defense (bool, optional): Defense flag. Defaults to False.
            squad_defense (bool, optional): Squad defense flag. Defaults to False.
        """
        if squad_defense:
            self.pokemon_info = self.pokemon_squad_info
        if defense:
            self.pokemon_info = self.pokemon_defense_info

        body_list = [
            dedent(default_messages["pokemon_info_template"]).format(
                name=name,
                id=pokemon["id"],
                hp=pokemon["stats"]["hp"],
                attack=pokemon["stats"]["attack"],
                defense=pokemon["stats"]["defense"],
                special_attack=pokemon["stats"]["special-attack"],
                special_defense=pokemon["stats"]["special-defense"],
                speed=pokemon["stats"]["speed"],
                height=pokemon["height"] / 10,
                weight=pokemon["weight"] / 10,
                types=", ".join(pokemon["types"]),
                abilities=", ".join(pokemon["abilities"]),
                damage_relations="\n- ".join(
                    [
                        f'{damage.replace("_", " ")}: {value}'
                        for damage, value in pokemon["damage_relations"].items()
                    ]
                ),
            )
            for name, pokemon in self.pokemon_info.items()
        ]
        sprite_list = {
            name: [url for url in value["sprites"].values() if url]
            for name, value in self.pokemon_info.items()
        }
        self.response["body"] = body_list
        self.response["sprites"] = sprite_list
