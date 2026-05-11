import pytest
from unittest.mock import AsyncMock
from crucible.integrations.langchain_adapter import LangChainAdapter


@pytest.mark.asyncio
async def test_langchain_adapter_success():
    # 1. Create a mock LangChain agent
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {"output": "Mocked success!"}

    # 2. Initialize your adapter
    adapter = LangChainAdapter(agent=mock_agent)

    # 3. Test the invoke method
    result = await adapter.invoke("Test prompt")

    # 4. Verify the results
    assert result["status"] == "success"
    assert result["response"] == "Mocked success!"
    mock_agent.ainvoke.assert_called_once_with({"input": "Test prompt"})
