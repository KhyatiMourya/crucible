import logging
from typing import Any

logger = logging.getLogger(__name__)


class LangChainAdapter:
    def __init__(self, agent: Any):
        """
        Initialize the adapter with a compiled LangChain agent.
        """
        self.agent = agent

    async def invoke(self, prompt: str) -> dict[str, Any]:
        """
        Directly invoke the LangChain agent without HTTP.
        """
        try:
            # LangChain's async invoke method
            response = await self.agent.ainvoke({"input": prompt})

            # Extract the actual string output from LangChain's response dictionary
            output_text = response.get("output", str(response))

            return {"status": "success", "response": output_text}
        except Exception as e:
            logger.error(f"LangChain direct invocation failed: {e}")
            return {"status": "error", "error_message": str(e)}
