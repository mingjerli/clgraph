"""Integration tests: prompt-injection defenses reached through public LLM tools.

Unit tests in test_prompt_sanitization.py cover the sanitizers directly.
These verify the defenses are actually wired into the call sites.
"""

from clgraph.tools.base import LLMTool


class _CaptureLLM:
    """Records how it was invoked so tests can assert on message structure."""

    def __init__(self):
        self.last_invocation = None

    def invoke(self, value):
        self.last_invocation = value

        class _Resp:
            content = "ok"

        return _Resp()


class _MiniTool(LLMTool):
    # BaseTool's actual abstract surface is `parameters` (property) and
    # `run()`, not `get_schema`/`execute` — stubbed here to match
    # src/clgraph/tools/base.py so the class can be instantiated.
    @property
    def parameters(self):
        return {}

    def run(self, *args, **kwargs):
        return None


def test_call_llm_structured_separates_system_and_user():
    llm = _CaptureLLM()
    tool = _MiniTool.__new__(_MiniTool)
    tool.llm = llm
    result = tool.call_llm_structured("You are X.", "raw user data")
    assert result == "ok"
    messages = llm.last_invocation
    assert len(messages) == 2
    assert messages[0].content == "You are X."
    assert messages[1].content == "raw user data"

    from langchain_core.messages import HumanMessage, SystemMessage

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
