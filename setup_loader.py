import logging
import os
import openai
from conf.config_loader import global_conf, prompt_template_library
from langchain_openai import ChatOpenAI
from agents.callbacks_agent import AgentCallbackHandler
from dotenv import load_dotenv


class SetupLoader:
    """ -- Singleton design pattern to instantiate the application --
    Validation: Ensures that subsequent calls to SetupLoader() will return the same
    instance configuration.
    """

    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SetupLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, new_model=False):
        load_dotenv()
        if not self.__class__._is_initialized:  # Ensure __init__ is only executed once
            self.logger = self._setup_logging()
            self._setup_environment()
            self.prompt_template_library = self._setup_prompt_library()
            self.global_conf = self._setup_global_conf()
            self.chat_openai = self._setup_chat_openai(
                callbacks=self._setup_callbacks()
            )
            self.__class__._is_initialized = True
        elif new_model:
            # After the first initialization, a new model can be created
            self.chat_openai = self._setup_chat_openai(
                callbacks=self._setup_callbacks()
            )

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _setup_prompt_library(self):
        return prompt_template_library

    def _setup_global_conf(self):
        return global_conf

    def _setup_callbacks(self):
        return [AgentCallbackHandler()]

    def _setup_environment(self):
        """Set the OpenAI API key from global_conf.yml if the user defined it."""
        if global_conf.get("OPENAI_API_KEY", None):
            os.environ["OPENAI_API_KEY"] = global_conf["OPENAI_API_KEY"]

    def _setup_chat_openai(self, callbacks=None):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        return ChatOpenAI(
            temperature=global_conf["MODEL_CREATIVITY"],
            model_name=global_conf["MODEL_NAME"],
            callbacks=callbacks,
        )
