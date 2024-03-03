from typing import Dict, Any, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for the agent."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running.
        Args:
            serialized (Dict[str, Any]): Serialized data.
            prompts (List[str]): List of prompts.
        """
        print("---------")
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("---------")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running.
        Args:
            response (LLMResult): LLM response.
        """
        print("---------")
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("---------")
