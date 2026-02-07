"""
Path validation module for secure file operations.

This module provides security-focused path validation to prevent:
- Path traversal attacks (../../../etc/passwd)
- Symlink attacks
- TOCTOU (Time-Of-Check-Time-Of-Use) vulnerabilities
- Windows reserved name attacks
- Unicode normalization attacks

Usage:
    from clgraph.path_validation import PathValidator, _safe_read_sql_file

    validator = PathValidator()
    validated_dir = validator.validate_directory("/path/to/sql/files")
    pattern = validator.validate_glob_pattern("*.sql", allowed_extensions=[".sql"])

    for sql_file in validated_dir.glob(pattern):
        content = _safe_read_sql_file(sql_file, base_dir=validated_dir)
"""

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# Windows reserved device names (case-insensitive)
WINDOWS_RESERVED_NAMES = frozenset(
    [
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    ]
)


class PathValidator:
    """Validates file and directory paths for security.

    This class provides methods to validate paths against common attack vectors
    including path traversal, symlink attacks, and Windows reserved names.

    Example:
        validator = PathValidator()
        validated_dir = validator.validate_directory("/data/queries")
        pattern = validator.validate_glob_pattern("*.sql", allowed_extensions=[".sql"])
    """

    def validate_directory(
        self,
        path: Union[str, Path],
        allow_symlinks: bool = False,
    ) -> Path:
        """Validate a directory path for security.

        Args:
            path: The directory path to validate (string or Path object).
            allow_symlinks: If True, symlinks are allowed. Defaults to False.
                           A security warning is logged when True.

        Returns:
            The resolved, validated Path object.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If path contains traversal sequences, is not a directory,
                       or is a symlink (when allow_symlinks=False).
            TypeError: If path is None or invalid type.
        """
        if path is None:
            raise TypeError("Path cannot be None")

        # Convert to string and handle empty path
        path_str = str(path).strip()
        if not path_str:
            raise ValueError("Path cannot be empty")

        # Apply NFKC normalization to detect Unicode tricks
        path_str = self._normalize_path(path_str)

        # Expand tilde
        path_str = os.path.expanduser(path_str)

        # Convert to Path object
        path_obj = Path(path_str)

        # Detect path traversal BEFORE resolution
        self._check_traversal_in_path(path_str)

        # Resolve to absolute path
        try:
            resolved = path_obj.resolve()
        except OSError as e:
            raise ValueError(f"Invalid path: {e}") from e

        # Check if path exists
        if not resolved.exists():
            raise FileNotFoundError("Directory does not exist: path not found")

        # Check if it's a directory
        if not resolved.is_dir():
            raise ValueError("Path is not a directory")

        # Check for symlinks
        if path_obj.is_symlink() and not allow_symlinks:
            raise ValueError("Symbolic links are not allowed (use allow_symlinks=True to override)")

        # Log warning when symlinks are allowed
        if allow_symlinks and path_obj.is_symlink():
            logger.warning(
                "SECURITY: allow_symlinks=True enables following symbolic links. "
                "This may expose sensitive files outside the intended directory."
            )

        return resolved

    def validate_file(
        self,
        path: Union[str, Path],
        allowed_extensions: List[str],
        allow_symlinks: bool = False,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Validate a file path for security.

        Args:
            path: The file path to validate.
            allowed_extensions: List of allowed file extensions (e.g., [".sql", ".json"]).
            allow_symlinks: If True, symlinks are allowed. Defaults to False.
            base_dir: If provided, the file must be within this directory.

        Returns:
            The resolved, validated Path object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If path contains traversal sequences, has wrong extension,
                       is not a file, is outside base_dir, or is a symlink
                       (when allow_symlinks=False).
            TypeError: If path is None or invalid type.
        """
        if path is None:
            raise TypeError("Path cannot be None")

        # Convert to string and handle empty path
        path_str = str(path).strip()
        if not path_str:
            raise ValueError("Path cannot be empty")

        # Apply NFKC normalization
        path_str = self._normalize_path(path_str)

        # Expand tilde
        path_str = os.path.expanduser(path_str)

        # Convert to Path object
        path_obj = Path(path_str)

        # Detect path traversal BEFORE resolution
        self._check_traversal_in_path(path_str)

        # Resolve to absolute path
        try:
            resolved = path_obj.resolve()
        except OSError as e:
            raise ValueError(f"Invalid path: {e}") from e

        # Check if path exists
        if not resolved.exists():
            raise FileNotFoundError("File does not exist: path not found")

        # Check if it's a file
        if not resolved.is_file():
            raise ValueError("Path is not a file")

        # Check extension (case-insensitive)
        file_ext = resolved.suffix.lower()
        allowed_exts_lower = [ext.lower() for ext in allowed_extensions]
        if file_ext not in allowed_exts_lower:
            raise ValueError(f"Invalid file extension: expected one of {allowed_extensions}")

        # Check Windows reserved names
        self._check_windows_reserved_name(resolved.stem)

        # Check for symlinks
        if path_obj.is_symlink() and not allow_symlinks:
            raise ValueError("Symbolic links are not allowed (use allow_symlinks=True to override)")

        # Check if file is within base_dir
        if base_dir is not None:
            base_path = Path(base_dir).resolve()
            if not resolved.is_relative_to(base_path):
                raise ValueError("Path escapes the base directory")

        # Log warning when symlinks are allowed
        if allow_symlinks and path_obj.is_symlink():
            logger.warning(
                "SECURITY: allow_symlinks=True enables following symbolic links. "
                "This may expose sensitive files."
            )

        return resolved

    def validate_glob_pattern(
        self,
        pattern: str,
        allowed_extensions: List[str],
    ) -> str:
        """Validate a glob pattern for security.

        Args:
            pattern: The glob pattern to validate (e.g., "*.sql", "**/*.sql").
            allowed_extensions: List of allowed file extensions.

        Returns:
            The validated pattern (unchanged if valid).

        Raises:
            ValueError: If pattern contains traversal sequences, is empty,
                       or has an invalid extension.
        """
        if not pattern or not pattern.strip():
            raise ValueError("Glob pattern cannot be empty")

        pattern = pattern.strip()

        # Check for path traversal in pattern
        if ".." in pattern:
            raise ValueError("Glob pattern must not contain directory traversal components")

        # Check extension in pattern
        # Allow patterns like "*.sql", "**/*.sql", "subdir/*.sql"
        # Also allow "*" and "**/*" for all files
        if pattern not in ("*", "**/*"):
            # Extract extension from pattern
            pattern_ext = self._extract_pattern_extension(pattern)
            if pattern_ext:
                allowed_exts_lower = [ext.lower() for ext in allowed_extensions]
                if pattern_ext.lower() not in allowed_exts_lower:
                    raise ValueError(
                        f"Invalid extension in pattern: expected one of {allowed_extensions}"
                    )

        return pattern

    def _normalize_path(self, path_str: str) -> str:
        """Apply Unicode NFKC normalization to path.

        This helps detect homoglyph attacks and Unicode escape sequences.
        """
        return unicodedata.normalize("NFKC", path_str)

    def _check_traversal_in_path(self, path_str: str) -> None:
        """Check for path traversal sequences in the path string.

        Raises:
            ValueError: If path contains traversal sequences.
        """
        # Normalize path separators
        normalized = path_str.replace("\\", "/")

        # Split and check each component
        components = normalized.split("/")

        for component in components:
            if component == "..":
                raise ValueError("Path traversal detected: path contains '..' component")

        # Note: Fullwidth periods (U+FF0E) are converted to regular periods by
        # NFKC normalization in _normalize_path() before this function is called,
        # so they are caught by the ".." check above.

    def _check_windows_reserved_name(self, name: str) -> None:
        """Check if name is a Windows reserved device name.

        Args:
            name: The filename stem (without extension) to check.

        Raises:
            ValueError: If name is a Windows reserved name.
        """
        if self._is_windows_reserved_name(name):
            raise ValueError(f"Reserved Windows device name not allowed: {name}")

    def _is_windows_reserved_name(self, name: str) -> bool:
        """Check if a name is a Windows reserved device name.

        Args:
            name: The name to check (case-insensitive).

        Returns:
            True if the name is reserved, False otherwise.
        """
        if not name:
            return False
        return name.upper() in WINDOWS_RESERVED_NAMES

    def _extract_pattern_extension(self, pattern: str) -> Optional[str]:
        """Extract the file extension from a glob pattern.

        Args:
            pattern: The glob pattern (e.g., "*.sql", "**/*.sql").

        Returns:
            The extension (e.g., ".sql") or None if no extension found.
        """
        # Handle patterns like "*.sql", "**/*.sql", "subdir/*.sql"
        # Get the last component
        parts = pattern.replace("\\", "/").split("/")
        last_part = parts[-1]

        # Check for extension pattern
        if "." in last_part and not last_part.startswith("."):
            # Extract extension (e.g., "*.sql" -> ".sql")
            ext_match = re.search(r"\.[a-zA-Z0-9]+$", last_part)
            if ext_match:
                return ext_match.group(0)

        return None


def _safe_read_sql_file(
    path: Union[str, Path],
    base_dir: Union[str, Path],
    allow_symlinks: bool = False,
) -> str:
    """Read SQL file with validation at read time to prevent TOCTOU attacks.

    This function validates the path immediately before reading, eliminating
    the race window between validation and read that could be exploited.

    Args:
        path: Path to the SQL file to read.
        base_dir: The base directory that the file must be within.
        allow_symlinks: If True, symlinks are allowed. Defaults to False.

    Returns:
        The contents of the SQL file as a string.

    Raises:
        ValueError: For security violations (path traversal, symlinks, wrong extension).
        FileNotFoundError: If file does not exist.
        PermissionError: If file cannot be read (re-raised with safe message).
    """
    path_obj = Path(path)
    base_path = Path(base_dir).resolve()

    # Re-resolve immediately before read to catch TOCTOU attacks
    try:
        resolved = path_obj.resolve()
    except OSError as e:
        raise ValueError("Invalid path") from e

    # Check confinement
    if not resolved.is_relative_to(base_path):
        raise ValueError("Path escapes the base directory")

    # Check symlink policy at read time (check the original path, not resolved)
    if path_obj.is_symlink() and not allow_symlinks:
        raise ValueError("Symbolic links are not allowed (use allow_symlinks=True to override)")

    # Validate extension
    if resolved.suffix.lower() != ".sql":
        raise ValueError("Invalid file extension: expected .sql")

    # Check if file exists
    if not resolved.exists():
        raise FileNotFoundError("SQL file not found")

    # Read with error handling that doesn't leak path info
    try:
        return resolved.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError("SQL file not found") from e
    except PermissionError as e:
        raise PermissionError("Cannot read SQL file: permission denied") from e
    except UnicodeDecodeError as e:
        raise ValueError("SQL file is not valid UTF-8") from e
    except OSError as e:
        raise ValueError(f"Cannot read SQL file: {type(e).__name__}") from e
