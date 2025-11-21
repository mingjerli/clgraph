"""
Parse metadata from SQL inline comments.

Extracts structured metadata from SQL comments in the format:
  <description> [key: value, key2: value2, ...]

Examples:
  -- User email address [pii: true, owner: data-team]
  /* Total revenue [tags: metric, finance] */
  -- Simple description without metadata
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from sqlglot import exp


@dataclass
class ColumnMetadata:
    """Extracted metadata from SQL comments."""

    description: Optional[str] = None
    pii: Optional[bool] = None
    owner: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "ColumnMetadata") -> "ColumnMetadata":
        """
        Merge another ColumnMetadata into this one.

        Rules:
        - First value wins for description, pii, owner
        - Tags are unioned
        - Custom metadata: first value wins per key
        """
        return ColumnMetadata(
            description=self.description or other.description,
            pii=self.pii if self.pii is not None else other.pii,
            owner=self.owner or other.owner,
            tags=self.tags | other.tags,
            custom_metadata={**other.custom_metadata, **self.custom_metadata},
        )


class MetadataExtractor:
    """Extract metadata from SQL AST nodes."""

    # Regex pattern for structured metadata: [key: value, key2: value2]
    METADATA_PATTERN = re.compile(r"\[(.*?)\]")

    def extract_from_expression(self, expr: exp.Expression) -> ColumnMetadata:
        """
        Extract metadata from SQL expression inline comments.

        Handles both -- and /* */ comment styles:
          col_name,  -- Description [pii: true, owner: team]
          SUM(amount) as total  /* Total amount [owner: finance] */

        Args:
            expr: sqlglot expression node

        Returns:
            ColumnMetadata with extracted metadata
        """
        # Check if expression has comments
        if not hasattr(expr, "comments") or not expr.comments:
            return ColumnMetadata()

        # Process all comments (usually just one)
        all_metadata = []
        for comment in expr.comments:
            metadata_dict = self._parse_comment_metadata(comment)
            if metadata_dict:
                all_metadata.append(self._dict_to_metadata(metadata_dict))

        # Merge metadata from multiple comments (if any)
        if not all_metadata:
            return ColumnMetadata()

        result = all_metadata[0]
        for metadata in all_metadata[1:]:
            result = result.merge(metadata)

        return result

    def _parse_comment_metadata(self, comment: str) -> Dict[str, Any]:
        """
        Parse structured metadata from comment string.

        Supports formats:
        - "Description only"
        - "Description [key: value, key2: value2]"
        - "[key: value] only"

        Args:
            comment: Raw comment string (may include leading/trailing whitespace)

        Returns:
            Dict with 'description' and parsed key-value pairs
        """
        comment = comment.strip()
        if not comment:
            return {}

        metadata = {}

        # Check for structured metadata in brackets
        match = self.METADATA_PATTERN.search(comment)

        if match:
            # Extract description (text before brackets)
            description = comment[: match.start()].strip()
            if description:
                metadata["description"] = description

            # Parse key-value pairs inside brackets
            metadata_str = match.group(1)
            parsed_pairs = self._parse_key_value_pairs(metadata_str)
            metadata.update(parsed_pairs)
        else:
            # No structured metadata, treat entire comment as description
            metadata["description"] = comment

        return metadata

    def _parse_key_value_pairs(self, metadata_str: str) -> Dict[str, Any]:
        """
        Parse key-value pairs from metadata string.

        Format: "key: value, key2: value2, key3: value3"

        Supports:
        - Boolean values: true, false (case-insensitive)
        - Integer values: 123
        - String values: anything else
        - Tags: Space-separated values for 'tags' key

        Args:
            metadata_str: String containing key-value pairs

        Returns:
            Dict with parsed key-value pairs (keys normalized to lowercase)
        """
        result = {}

        # Split by comma, but be careful with nested structures
        pairs = [p.strip() for p in metadata_str.split(",")]

        for pair in pairs:
            if ":" not in pair:
                # Malformed pair, skip it
                continue

            key, value = pair.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            # Type conversion
            if value.lower() in ("true", "false"):
                result[key] = value.lower() == "true"
            elif value.isdigit():
                result[key] = int(value)
            else:
                # Handle tags specially (space-separated or comma-separated)
                if key == "tags":
                    # Split by spaces to handle multiple tags
                    tag_values = {t.strip() for t in value.split() if t.strip()}
                    result[key] = tag_values
                else:
                    result[key] = value

        return result

    def _dict_to_metadata(self, metadata_dict: Dict[str, Any]) -> ColumnMetadata:
        """
        Convert parsed metadata dict to ColumnMetadata object.

        Args:
            metadata_dict: Dict with parsed metadata

        Returns:
            ColumnMetadata instance
        """
        return ColumnMetadata(
            description=metadata_dict.get("description"),
            pii=metadata_dict.get("pii"),
            owner=metadata_dict.get("owner"),
            tags=metadata_dict.get("tags", set()),
            custom_metadata={
                k: v
                for k, v in metadata_dict.items()
                if k not in ["description", "pii", "owner", "tags"]
            },
        )


__all__ = [
    "ColumnMetadata",
    "MetadataExtractor",
]
