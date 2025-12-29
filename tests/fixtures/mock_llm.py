"""
Mock LLM for testing README examples.

Provides predictable responses for agent and tool tests without
requiring an actual LLM connection.
"""


class MockLLMResponse:
    """Mock response object that mimics LangChain response structure."""

    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """
    Mock LLM that returns predictable responses for testing.

    This allows README examples to be tested without requiring
    an actual LLM API connection.
    """

    def __init__(self, model: str = "mock-model", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature

    def invoke(self, messages) -> MockLLMResponse:
        """
        Return mock responses based on input patterns.

        For testing purposes, returns sensible default responses
        that match what real LLMs would return.
        """
        # Handle string input
        if isinstance(messages, str):
            prompt = messages.lower()
        # Handle list of messages (LangChain format)
        elif isinstance(messages, list):
            # Combine all message contents
            prompt = " ".join(
                m.content.lower() if hasattr(m, "content") else str(m).lower() for m in messages
            )
        else:
            prompt = str(messages).lower()

        # SQL generation responses
        if "top 10 customers" in prompt and "lifetime" in prompt:
            return MockLLMResponse(
                "SELECT customer_id, email, lifetime_value\n"
                "FROM analytics.customers\n"
                "ORDER BY lifetime_value DESC\n"
                "LIMIT 10"
            )
        elif "monthly revenue" in prompt:
            return MockLLMResponse(
                "SELECT DATE_TRUNC(order_date, MONTH) as month, SUM(amount) as revenue\n"
                "FROM analytics.orders\n"
                "WHERE order_date >= '2024-01-01'\n"
                "GROUP BY 1\n"
                "ORDER BY 1"
            )

        # Description generation responses
        elif "description" in prompt or "describe" in prompt:
            return MockLLMResponse("This column contains business data used for analytics.")

        # Generic response
        return MockLLMResponse("Mock LLM response for testing.")
