"""
Base classes for lineage tools.

Provides the foundation for all tools in the lineage intelligence platform.
Tools are the universal primitive - everything else (Agent, MCP) builds on them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    from ..pipeline import Pipeline


class ParameterType(Enum):
    """Parameter types for tool definitions."""

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ParameterSpec:
    """Specification for a tool parameter."""

    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None  # For constrained string values

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format (for MCP/OpenAI function calling)."""
        schema: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolResult:
    """
    Standard result from any tool execution.

    All tools return this consistent format, making it easy for
    agents and MCP servers to handle results uniformly.
    """

    success: bool
    """Whether the tool executed successfully."""

    data: Any
    """Tool-specific structured data (list, dict, etc.)."""

    message: str
    """Human-readable summary of the result."""

    error: Optional[str] = None
    """Error message if success=False."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata (timing, debug info, etc.)."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization/MCP)."""
        result = {
            "success": self.success,
            "data": self._serialize_data(self.data),
            "message": self.message,
        }
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON compatibility."""
        if data is None:
            return None
        if isinstance(data, (str, int, float, bool)):
            return data
        if isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        if isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        if hasattr(data, "to_dict"):
            return data.to_dict()
        if hasattr(data, "__dict__"):
            # Handle dataclasses and similar
            return {
                k: self._serialize_data(v)
                for k, v in data.__dict__.items()
                if not k.startswith("_")
            }
        return str(data)

    @classmethod
    def success_result(cls, data: Any, message: str, **metadata) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, data=data, message=message, metadata=metadata)

    @classmethod
    def error_result(cls, error: str, **metadata) -> "ToolResult":
        """Create an error result."""
        return cls(
            success=False, data=None, message=f"Error: {error}", error=error, metadata=metadata
        )


class BaseTool(ABC):
    """
    Base class for all lineage tools.

    Tools are the fundamental building blocks of the lineage intelligence
    platform. Each tool performs a specific operation on a Pipeline.

    Subclasses must implement:
    - name: Unique identifier for the tool
    - description: Human-readable description
    - parameters: Dict of ParameterSpec defining inputs
    - run(): Execute the tool

    Example:
        class MyTool(BaseTool):
            name = "my_tool"
            description = "Does something useful"

            @property
            def parameters(self) -> Dict[str, ParameterSpec]:
                return {
                    "table": ParameterSpec("table", ParameterType.STRING, "Table name"),
                }

            def run(self, table: str) -> ToolResult:
                # Do something
                return ToolResult.success_result(data, "Success!")
    """

    name: str
    """Unique identifier for the tool."""

    description: str
    """Human-readable description of what the tool does."""

    def __init__(self, pipeline: "Pipeline"):
        """
        Initialize tool with a Pipeline.

        Args:
            pipeline: The clgraph Pipeline to operate on.
        """
        self.pipeline = pipeline

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, ParameterSpec]:
        """
        Define the tool's input parameters.

        Returns:
            Dict mapping parameter names to ParameterSpec definitions.
        """
        pass

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Parameters as defined by self.parameters

        Returns:
            ToolResult with success status, data, and message.
        """
        pass

    def validate_params(self, **kwargs) -> Optional[str]:
        """
        Validate input parameters.

        Returns:
            Error message if validation fails, None if valid.
        """
        for name, spec in self.parameters.items():
            if spec.required and name not in kwargs:
                return f"Missing required parameter: {name}"

            if name in kwargs:
                value = kwargs[name]
                # Type validation
                if spec.type == ParameterType.STRING and not isinstance(value, str):
                    return f"Parameter '{name}' must be a string"
                if spec.type == ParameterType.INTEGER and not isinstance(value, int):
                    return f"Parameter '{name}' must be an integer"
                if spec.type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    return f"Parameter '{name}' must be a boolean"
                # Enum validation
                if spec.enum and value not in spec.enum:
                    return f"Parameter '{name}' must be one of: {spec.enum}"

        return None

    def __call__(self, **kwargs) -> ToolResult:
        """Allow calling tool as function."""
        # Validate parameters
        error = self.validate_params(**kwargs)
        if error:
            return ToolResult.error_result(error)

        # Apply defaults
        for name, spec in self.parameters.items():
            if name not in kwargs and spec.default is not None:
                kwargs[name] = spec.default

        return self.run(**kwargs)

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert tool definition to JSON Schema format.

        Compatible with OpenAI function calling and MCP tool definitions.
        """
        properties = {}
        required = []

        for name, spec in self.parameters.items():
            properties[name] = spec.to_json_schema()
            if spec.required:
                required.append(name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class LLMTool(BaseTool):
    """
    Base class for tools that require an LLM.

    Extends BaseTool with LLM integration for tools like
    SQL generation that need language model capabilities.
    """

    def __init__(self, pipeline: "Pipeline", llm: Any):
        """
        Initialize tool with Pipeline and LLM.

        Args:
            pipeline: The clgraph Pipeline to operate on.
            llm: LangChain-compatible LLM or callable.
        """
        super().__init__(pipeline)
        self.llm = llm

    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM with a prompt.

        Handles different LLM interfaces (LangChain, callable).

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The LLM's response as a string.
        """
        if hasattr(self.llm, "invoke"):
            # LangChain interface
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                return response.content
            return str(response)
        elif callable(self.llm):
            # Simple callable
            return self.llm(prompt)
        else:
            raise ValueError(f"Unsupported LLM type: {type(self.llm)}")


class ToolRegistry:
    """
    Registry of all available tools.

    Provides a centralized way to manage and access tools,
    used by LineageAgent and MCP Server.
    """

    def __init__(self, pipeline: "Pipeline", llm: Any = None):
        """
        Initialize registry with Pipeline and optional LLM.

        Args:
            pipeline: The clgraph Pipeline for tool operations.
            llm: Optional LLM for tools that require it.
        """
        self.pipeline = pipeline
        self.llm = llm
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool_class: Type[BaseTool]) -> None:
        """
        Register a tool class.

        Args:
            tool_class: The tool class to register.
        """
        if issubclass(tool_class, LLMTool):
            if self.llm is None:
                # Skip LLM tools if no LLM provided
                return
            tool = tool_class(self.pipeline, self.llm)
        else:
            tool = tool_class(self.pipeline)

        self._tools[tool.name] = tool

    def register_all(self, tool_classes: List[Type[BaseTool]]) -> None:
        """Register multiple tool classes."""
        for tool_class in tool_classes:
            self.register(tool_class)

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def tool_names(self) -> List[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    def to_json_schema(self) -> List[Dict[str, Any]]:
        """Get JSON schema for all tools (for MCP/function calling)."""
        return [tool.to_json_schema() for tool in self._tools.values()]

    def run(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Run a tool by name.

        Args:
            tool_name: Name of the tool to run.
            **kwargs: Parameters for the tool.

        Returns:
            ToolResult from the tool execution.
        """
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult.error_result(f"Unknown tool: {tool_name}")
        return tool(**kwargs)


__all__ = [
    "ParameterType",
    "ParameterSpec",
    "ToolResult",
    "BaseTool",
    "LLMTool",
    "ToolRegistry",
]
