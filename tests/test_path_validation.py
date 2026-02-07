"""
Tests for path validation module.

This module tests the PathValidator class and _safe_read_sql_file function
to ensure proper security against path traversal, symlink attacks, TOCTOU
vulnerabilities, and other file system-based attacks.

TDD Approach: These tests are written FIRST before implementation.
"""

import os
import platform
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Import will fail until we implement the module (RED phase)
# We use a try/except to make the test file parseable before implementation
try:
    from clgraph.path_validation import (
        PathValidator,
        _safe_read_sql_file,
    )
except ImportError:
    PathValidator = None
    _safe_read_sql_file = None


# Skip all tests if module not implemented yet
pytestmark = pytest.mark.skipif(
    PathValidator is None,
    reason="path_validation module not implemented yet",
)


class TestPathValidatorValidateDirectory:
    """Tests for PathValidator.validate_directory() method."""

    def test_valid_directory_returns_resolved_path(self, tmp_path):
        """Test that a valid directory returns a resolved Path object."""
        validator = PathValidator()
        result = validator.validate_directory(str(tmp_path))

        assert isinstance(result, Path)
        assert result == tmp_path.resolve()
        assert result.is_dir()

    def test_nonexistent_directory_raises_error(self):
        """Test that a nonexistent directory raises FileNotFoundError."""
        validator = PathValidator()

        with pytest.raises(FileNotFoundError, match="does not exist"):
            validator.validate_directory("/nonexistent/path/to/directory")

    def test_file_instead_of_directory_raises_error(self, tmp_path):
        """Test that passing a file instead of directory raises ValueError."""
        test_file = tmp_path / "test.sql"
        test_file.write_text("SELECT 1")

        validator = PathValidator()

        with pytest.raises(ValueError, match="not a directory"):
            validator.validate_directory(str(test_file))

    def test_path_traversal_with_double_dots_raises_error(self, tmp_path):
        """Test that .. in path is detected and rejected."""
        validator = PathValidator()

        # Create a path with traversal attempt
        traversal_path = str(tmp_path / ".." / "etc")

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            validator.validate_directory(traversal_path)

    def test_symlink_rejected_by_default(self, tmp_path):
        """Test that symlinks are rejected when allow_symlinks=False."""
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        symlink_dir = tmp_path / "link"

        # Create symlink (skip on Windows if no privilege)
        try:
            symlink_dir.symlink_to(target_dir)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        validator = PathValidator()

        with pytest.raises(ValueError, match="[Ss]ymbolic link"):
            validator.validate_directory(str(symlink_dir))

    def test_symlink_accepted_when_allowed(self, tmp_path):
        """Test that symlinks are accepted when allow_symlinks=True."""
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        symlink_dir = tmp_path / "link"

        try:
            symlink_dir.symlink_to(target_dir)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        validator = PathValidator()
        result = validator.validate_directory(str(symlink_dir), allow_symlinks=True)

        assert result.is_dir()

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        """Test that ~ is expanded to home directory."""
        # Create a test directory in home
        monkeypatch.setenv("HOME", str(tmp_path))

        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        validator = PathValidator()
        result = validator.validate_directory("~/test_dir")

        assert result == test_dir.resolve()

    def test_relative_path_resolved_to_absolute(self, tmp_path, monkeypatch):
        """Test that relative paths are resolved to absolute paths."""
        monkeypatch.chdir(tmp_path)

        test_dir = tmp_path / "subdir"
        test_dir.mkdir()

        validator = PathValidator()
        result = validator.validate_directory("./subdir")

        assert result.is_absolute()
        assert result == test_dir.resolve()


class TestPathValidatorValidateFile:
    """Tests for PathValidator.validate_file() method."""

    def test_valid_sql_file_returns_resolved_path(self, tmp_path):
        """Test that a valid .sql file returns a resolved Path object."""
        test_file = tmp_path / "query.sql"
        test_file.write_text("SELECT 1")

        validator = PathValidator()
        result = validator.validate_file(
            str(test_file),
            allowed_extensions=[".sql"],
        )

        assert isinstance(result, Path)
        assert result == test_file.resolve()

    def test_valid_json_file_returns_resolved_path(self, tmp_path):
        """Test that a valid .json file returns a resolved Path object."""
        test_file = tmp_path / "data.json"
        test_file.write_text("{}")

        validator = PathValidator()
        result = validator.validate_file(
            str(test_file),
            allowed_extensions=[".json"],
        )

        assert result == test_file.resolve()

    def test_nonexistent_file_raises_error(self):
        """Test that a nonexistent file raises FileNotFoundError."""
        validator = PathValidator()

        with pytest.raises(FileNotFoundError, match="does not exist"):
            validator.validate_file(
                "/nonexistent/file.sql",
                allowed_extensions=[".sql"],
            )

    def test_directory_instead_of_file_raises_error(self, tmp_path):
        """Test that passing a directory instead of file raises ValueError."""
        validator = PathValidator()

        with pytest.raises(ValueError, match="not a file"):
            validator.validate_file(
                str(tmp_path),
                allowed_extensions=[".sql"],
            )

    def test_wrong_extension_raises_error(self, tmp_path):
        """Test that a file with wrong extension raises ValueError."""
        test_file = tmp_path / "query.txt"
        test_file.write_text("SELECT 1")

        validator = PathValidator()

        with pytest.raises(ValueError, match="[Ii]nvalid.*extension"):
            validator.validate_file(
                str(test_file),
                allowed_extensions=[".sql"],
            )

    def test_multiple_allowed_extensions(self, tmp_path):
        """Test that multiple extensions can be allowed."""
        sql_file = tmp_path / "query.sql"
        sql_file.write_text("SELECT 1")

        json_file = tmp_path / "data.json"
        json_file.write_text("{}")

        validator = PathValidator()

        # Both should pass
        validator.validate_file(str(sql_file), allowed_extensions=[".sql", ".json"])
        validator.validate_file(str(json_file), allowed_extensions=[".sql", ".json"])

    def test_path_traversal_detected(self, tmp_path):
        """Test that .. in file path is detected and rejected."""
        validator = PathValidator()

        traversal_path = str(tmp_path / ".." / "etc" / "passwd")

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            validator.validate_file(traversal_path, allowed_extensions=[".sql"])

    def test_file_outside_base_dir_rejected(self, tmp_path):
        """Test that files outside base_dir are rejected."""
        # Create two separate directories
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        forbidden_dir = tmp_path / "forbidden"
        forbidden_dir.mkdir()

        forbidden_file = forbidden_dir / "secret.sql"
        forbidden_file.write_text("SELECT secret")

        validator = PathValidator()

        with pytest.raises(ValueError, match="[Ee]scapes.*base.*directory"):
            validator.validate_file(
                str(forbidden_file),
                allowed_extensions=[".sql"],
                base_dir=allowed_dir,
            )

    def test_file_inside_base_dir_accepted(self, tmp_path):
        """Test that files inside base_dir are accepted."""
        base_dir = tmp_path / "queries"
        base_dir.mkdir()

        test_file = base_dir / "query.sql"
        test_file.write_text("SELECT 1")

        validator = PathValidator()
        result = validator.validate_file(
            str(test_file),
            allowed_extensions=[".sql"],
            base_dir=base_dir,
        )

        assert result == test_file.resolve()

    def test_symlink_rejected_by_default(self, tmp_path):
        """Test that symlink to file is rejected by default."""
        target_file = tmp_path / "target.sql"
        target_file.write_text("SELECT 1")
        symlink_file = tmp_path / "link.sql"

        try:
            symlink_file.symlink_to(target_file)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        validator = PathValidator()

        with pytest.raises(ValueError, match="[Ss]ymbolic link"):
            validator.validate_file(str(symlink_file), allowed_extensions=[".sql"])

    def test_symlink_accepted_when_allowed(self, tmp_path):
        """Test that symlink is accepted when allow_symlinks=True."""
        target_file = tmp_path / "target.sql"
        target_file.write_text("SELECT 1")
        symlink_file = tmp_path / "link.sql"

        try:
            symlink_file.symlink_to(target_file)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        validator = PathValidator()
        result = validator.validate_file(
            str(symlink_file),
            allowed_extensions=[".sql"],
            allow_symlinks=True,
        )

        assert result.is_file()

    def test_case_insensitive_extension_matching(self, tmp_path):
        """Test that extension matching is case-insensitive."""
        test_file = tmp_path / "query.SQL"
        test_file.write_text("SELECT 1")

        validator = PathValidator()
        result = validator.validate_file(str(test_file), allowed_extensions=[".sql"])

        assert result == test_file.resolve()


class TestPathValidatorValidateGlobPattern:
    """Tests for PathValidator.validate_glob_pattern() method."""

    def test_valid_pattern_returns_unchanged(self):
        """Test that a valid pattern is returned unchanged."""
        validator = PathValidator()
        result = validator.validate_glob_pattern("*.sql", allowed_extensions=[".sql"])

        assert result == "*.sql"

    def test_recursive_pattern_accepted(self):
        """Test that recursive glob patterns are accepted."""
        validator = PathValidator()
        result = validator.validate_glob_pattern("**/*.sql", allowed_extensions=[".sql"])

        assert result == "**/*.sql"

    def test_pattern_with_directory_accepted(self):
        """Test that patterns with directory prefixes are accepted."""
        validator = PathValidator()
        result = validator.validate_glob_pattern(
            "subdir/*.sql",
            allowed_extensions=[".sql"],
        )

        assert result == "subdir/*.sql"

    def test_pattern_with_traversal_rejected(self):
        """Test that patterns containing .. are rejected."""
        validator = PathValidator()

        with pytest.raises(ValueError, match="[Tt]raversal"):
            validator.validate_glob_pattern("../*.sql", allowed_extensions=[".sql"])

    def test_pattern_with_hidden_traversal_rejected(self):
        """Test that patterns with traversal in middle are rejected."""
        validator = PathValidator()

        with pytest.raises(ValueError, match="[Tt]raversal"):
            validator.validate_glob_pattern(
                "subdir/../../../etc/*.sql",
                allowed_extensions=[".sql"],
            )

    def test_pattern_with_wrong_extension_rejected(self):
        """Test that patterns with wrong extension are rejected."""
        validator = PathValidator()

        with pytest.raises(ValueError, match="[Ii]nvalid.*extension"):
            validator.validate_glob_pattern("*.txt", allowed_extensions=[".sql"])

    def test_pattern_with_multiple_extensions(self):
        """Test patterns when multiple extensions are allowed."""
        validator = PathValidator()

        # Should pass
        result = validator.validate_glob_pattern(
            "*.sql",
            allowed_extensions=[".sql", ".json"],
        )
        assert result == "*.sql"

    def test_star_pattern_accepted(self):
        """Test that bare * pattern is accepted (matches all files)."""
        validator = PathValidator()

        # When using *, extension checking should pass if any extension is allowed
        result = validator.validate_glob_pattern("*", allowed_extensions=[".sql"])
        assert result == "*"

    def test_empty_pattern_rejected(self):
        """Test that empty pattern is rejected."""
        validator = PathValidator()

        with pytest.raises(ValueError, match="[Ee]mpty|[Ii]nvalid"):
            validator.validate_glob_pattern("", allowed_extensions=[".sql"])


class TestSafeReadSqlFile:
    """Tests for _safe_read_sql_file() function with TOCTOU mitigation."""

    def test_valid_file_returns_content(self, tmp_path):
        """Test that valid SQL file content is returned."""
        test_file = tmp_path / "query.sql"
        test_file.write_text("SELECT 1 FROM table")

        content = _safe_read_sql_file(test_file, base_dir=tmp_path)

        assert content == "SELECT 1 FROM table"

    def test_file_outside_base_dir_rejected(self, tmp_path):
        """Test that files outside base_dir are rejected at read time."""
        outside_dir = tmp_path.parent / "outside"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "secret.sql"
        outside_file.write_text("SECRET DATA")

        with pytest.raises(ValueError, match="[Ee]scapes.*base.*directory"):
            _safe_read_sql_file(outside_file, base_dir=tmp_path)

    def test_wrong_extension_rejected(self, tmp_path):
        """Test that wrong extension is rejected at read time."""
        test_file = tmp_path / "query.txt"
        test_file.write_text("SELECT 1")

        with pytest.raises(ValueError, match="[Ii]nvalid.*extension"):
            _safe_read_sql_file(test_file, base_dir=tmp_path)

    def test_symlink_rejected_by_default(self, tmp_path):
        """Test that symlinks are rejected at read time."""
        target = tmp_path / "target.sql"
        target.write_text("SELECT 1")
        link = tmp_path / "link.sql"

        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        with pytest.raises(ValueError, match="[Ss]ymbolic link"):
            _safe_read_sql_file(link, base_dir=tmp_path)

    def test_symlink_accepted_when_allowed(self, tmp_path):
        """Test that symlinks work when allow_symlinks=True."""
        target = tmp_path / "target.sql"
        target.write_text("SELECT 1")
        link = tmp_path / "link.sql"

        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        content = _safe_read_sql_file(link, base_dir=tmp_path, allow_symlinks=True)
        assert content == "SELECT 1"

    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        """Test that nonexistent file raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.sql"

        with pytest.raises(FileNotFoundError, match="not found"):
            _safe_read_sql_file(nonexistent, base_dir=tmp_path)

    def test_permission_error_handled(self, tmp_path):
        """Test that permission errors are handled safely."""
        test_file = tmp_path / "protected.sql"
        test_file.write_text("SELECT 1")

        # Skip on Windows (chmod doesn't work the same way)
        if platform.system() == "Windows":
            pytest.skip("Permission test not reliable on Windows")

        # Remove read permissions
        original_mode = test_file.stat().st_mode
        try:
            os.chmod(test_file, 0o000)

            with pytest.raises(PermissionError, match="permission denied"):
                _safe_read_sql_file(test_file, base_dir=tmp_path)
        finally:
            os.chmod(test_file, original_mode)

    def test_unicode_decode_error_handled(self, tmp_path):
        """Test that non-UTF8 files are handled with a clear error."""
        test_file = tmp_path / "binary.sql"
        # Write invalid UTF-8 bytes
        test_file.write_bytes(b"\xff\xfe SELECT 1")

        with pytest.raises(ValueError, match="[Nn]ot valid UTF-8"):
            _safe_read_sql_file(test_file, base_dir=tmp_path)

    def test_toctou_symlink_attack_simulation(self, tmp_path):
        """Test TOCTOU mitigation: file replaced with symlink between check and read.

        This simulates an attacker replacing a valid file with a symlink
        after the initial glob/validation but before the actual read.
        """
        # Setup: valid SQL file
        valid_file = tmp_path / "query.sql"
        valid_file.write_text("SELECT 1")

        # Create a "secret" file outside the directory
        outside_dir = tmp_path.parent / "secrets"
        outside_dir.mkdir(exist_ok=True)
        secret_file = outside_dir / "passwd"
        secret_file.write_text("secret password")

        # The validation should catch this at read time
        # because we re-validate immediately before reading
        def attack_thread():
            """Simulate attacker replacing file with symlink."""
            time.sleep(0.01)  # Small delay to simulate race
            try:
                valid_file.unlink()
                valid_file.symlink_to(secret_file)
            except (OSError, PermissionError):
                pass  # Symlink may fail on some systems

        thread = threading.Thread(target=attack_thread)
        thread.start()

        # Try reading - should fail if symlink is created
        try:
            # The function should validate at read time
            _safe_read_sql_file(valid_file, base_dir=tmp_path)
            # If we got here, either:
            # 1. The race didn't happen (we read before replacement)
            # 2. The symlink creation failed
            # Both are acceptable outcomes
            if valid_file.is_symlink():
                # If it's a symlink now, we should have been rejected
                pytest.fail("TOCTOU vulnerability: symlink was not detected")
        except ValueError as e:
            # Expected: symlink or path traversal detected
            assert "symlink" in str(e).lower() or "escapes" in str(e).lower()
        except FileNotFoundError:
            # File was deleted during race - acceptable
            pass
        finally:
            thread.join()


class TestWindowsSpecificValidation:
    """Tests for Windows-specific path validation."""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only test")
    def test_reserved_names_rejected(self, tmp_path):
        """Test that Windows reserved names are rejected."""
        validator = PathValidator()

        reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "LPT1", "LPT2"]

        for name in reserved_names:
            with pytest.raises(ValueError, match="[Rr]eserved"):
                validator.validate_file(
                    str(tmp_path / f"{name}.sql"),
                    allowed_extensions=[".sql"],
                )

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only test")
    def test_reserved_names_case_insensitive(self, tmp_path):
        """Test that Windows reserved name check is case-insensitive."""
        validator = PathValidator()

        # These should all be rejected
        for name in ["con", "Con", "CON", "CoN"]:
            with pytest.raises(ValueError, match="[Rr]eserved"):
                validator.validate_file(
                    str(tmp_path / f"{name}.sql"),
                    allowed_extensions=[".sql"],
                )

    def test_windows_reserved_names_check_on_all_platforms(self):
        """Test that Windows reserved names are checked even on non-Windows.

        This ensures portability - files created on Linux that would be
        problematic on Windows are caught early.
        """
        validator = PathValidator()

        # The validator should have a method to check Windows reserved names
        assert hasattr(validator, "_is_windows_reserved_name")

        # These should all be detected as reserved
        assert validator._is_windows_reserved_name("CON") is True
        assert validator._is_windows_reserved_name("con") is True
        assert validator._is_windows_reserved_name("PRN") is True
        assert validator._is_windows_reserved_name("NUL") is True
        assert validator._is_windows_reserved_name("COM1") is True
        assert validator._is_windows_reserved_name("LPT1") is True

        # These should not be reserved
        assert validator._is_windows_reserved_name("query") is False
        assert validator._is_windows_reserved_name("connector") is False
        assert validator._is_windows_reserved_name("console") is False


class TestUnicodeNormalization:
    """Tests for Unicode normalization attack prevention."""

    def test_unicode_path_traversal_detected(self, tmp_path):
        """Test that Unicode-encoded .. sequences are detected."""
        validator = PathValidator()

        # Various Unicode representations of ".."
        unicode_traversals = [
            # Fullwidth period and dot
            str(tmp_path / "\uff0e\uff0e" / "etc"),
            # Unicode escape sequences (if they get through somehow)
            str(tmp_path / "\u002e\u002e" / "etc"),
        ]

        for path in unicode_traversals:
            try:
                result = validator.validate_directory(path)
                # If it resolved, make sure it didn't escape
                if not str(result).startswith(str(tmp_path)):
                    pytest.fail(f"Unicode traversal escaped: {path}")
            except (ValueError, FileNotFoundError):
                # Expected - either caught as traversal or path doesn't exist
                pass

    def test_homoglyph_detection(self, tmp_path):
        """Test that homoglyph attacks in paths are handled.

        Homoglyphs are characters that look similar but have different
        Unicode code points (e.g., Cyrillic 'a' vs ASCII 'a').
        """
        validator = PathValidator()

        # Path with Cyrillic 'a' instead of ASCII 'a'
        # This creates a path that looks like "data" but isn't
        cyrillic_path = str(tmp_path / "d\u0430ta")  # Cyrillic 'a'

        # The validation should either:
        # 1. Normalize the path and find it doesn't exist
        # 2. Accept it if the actual path exists
        # The key is it shouldn't be confused with "data"
        try:
            validator.validate_directory(cyrillic_path)
            pytest.fail("Non-existent path with homoglyphs should not validate")
        except (FileNotFoundError, ValueError):
            pass  # Expected

    def test_nfkc_normalization_applied(self):
        """Test that NFKC normalization is applied to paths."""
        validator = PathValidator()

        # Fullwidth characters should be normalized
        # We can test the internal normalization if exposed
        if hasattr(validator, "_normalize_path"):
            # Fullwidth slash
            normalized = validator._normalize_path("/tmp\uff0ftest")
            assert "\uff0f" not in str(normalized)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_path_rejected(self):
        """Test that empty path is rejected."""
        validator = PathValidator()

        with pytest.raises((ValueError, FileNotFoundError)):
            validator.validate_directory("")

    def test_none_path_raises_type_error(self):
        """Test that None path raises appropriate error."""
        validator = PathValidator()

        with pytest.raises((TypeError, ValueError)):
            validator.validate_directory(None)

    def test_very_long_path_handled(self, tmp_path):
        """Test that very long paths are handled gracefully."""
        validator = PathValidator()

        # Create a path that exceeds typical limits
        long_component = "a" * 255  # Max filename length on most systems
        long_path = str(tmp_path / long_component)

        # Should raise an appropriate error, not crash
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            validator.validate_directory(long_path)

    def test_path_with_null_bytes_rejected(self, tmp_path):
        """Test that paths with null bytes are rejected."""
        validator = PathValidator()

        null_path = str(tmp_path) + "\x00/malicious"

        with pytest.raises((ValueError, TypeError, OSError)):
            validator.validate_directory(null_path)

    def test_path_with_special_characters(self, tmp_path):
        """Test that paths with valid special characters work."""
        # Create directory with spaces and special chars
        special_dir = tmp_path / "my queries (v2)"
        special_dir.mkdir()

        validator = PathValidator()
        result = validator.validate_directory(str(special_dir))

        assert result == special_dir.resolve()

    def test_nested_symlink_chain_rejected(self, tmp_path):
        """Test that chains of symlinks are all rejected."""
        target = tmp_path / "target"
        target.mkdir()

        link1 = tmp_path / "link1"
        link2 = tmp_path / "link2"

        try:
            link1.symlink_to(target)
            link2.symlink_to(link1)  # link2 -> link1 -> target
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        validator = PathValidator()

        with pytest.raises(ValueError, match="[Ss]ymbolic link"):
            validator.validate_directory(str(link2))


class TestIntegration:
    """Integration tests for path validation with actual file operations."""

    def test_full_roundtrip_sql_files(self, tmp_path):
        """Test complete workflow of validating and reading SQL files."""
        # Setup: create SQL files
        sql_dir = tmp_path / "queries"
        sql_dir.mkdir()

        (sql_dir / "01_staging.sql").write_text("CREATE TABLE staging AS SELECT 1")
        (sql_dir / "02_final.sql").write_text("CREATE TABLE final AS SELECT * FROM staging")

        # Validate directory
        validator = PathValidator()
        validated_dir = validator.validate_directory(str(sql_dir))

        # Validate pattern
        pattern = validator.validate_glob_pattern("*.sql", allowed_extensions=[".sql"])

        # Read files safely
        sql_files = sorted(validated_dir.glob(pattern))
        contents = []
        for sql_file in sql_files:
            content = _safe_read_sql_file(sql_file, base_dir=validated_dir)
            contents.append(content)

        assert len(contents) == 2
        assert "CREATE TABLE staging" in contents[0]
        assert "CREATE TABLE final" in contents[1]

    def test_subdirectory_traversal_blocked(self, tmp_path):
        """Test that traversal from subdirectory is blocked."""
        # Create structure:
        # tmp_path/
        #   queries/
        #     valid.sql
        #   secrets/
        #     password.sql
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "valid.sql").write_text("SELECT 1")

        secrets_dir = tmp_path / "secrets"
        secrets_dir.mkdir()
        (secrets_dir / "password.sql").write_text("SECRET")

        # Attempt to read secrets via traversal
        validator = PathValidator()
        validated_dir = validator.validate_directory(str(queries_dir))

        # Direct path traversal attempt
        secret_path = queries_dir / ".." / "secrets" / "password.sql"

        with pytest.raises(ValueError, match="[Ee]scapes.*base.*directory"):
            _safe_read_sql_file(secret_path.resolve(), base_dir=validated_dir)

    def test_recursive_glob_with_symlinks_blocked(self, tmp_path):
        """Test that recursive globs with symlinks are blocked."""
        # Create structure with symlink escape
        queries_dir = tmp_path / "queries"
        queries_dir.mkdir()
        (queries_dir / "valid.sql").write_text("SELECT 1")

        secrets_dir = tmp_path / "secrets"
        secrets_dir.mkdir()
        (secrets_dir / "password.sql").write_text("SECRET")

        # Create symlink in queries pointing to secrets
        escape_link = queries_dir / "escape"
        try:
            escape_link.symlink_to(secrets_dir)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        validator = PathValidator()
        validated_dir = validator.validate_directory(str(queries_dir))

        # Try to read via the symlink
        symlink_path = escape_link / "password.sql"

        # Should fail because the symlink target is outside base_dir
        with pytest.raises(ValueError):
            _safe_read_sql_file(symlink_path, base_dir=validated_dir)


class TestLogging:
    """Tests for security logging behavior."""

    def test_symlink_warning_logged(self, tmp_path, caplog):
        """Test that using allow_symlinks=True logs a warning."""
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        symlink_dir = tmp_path / "link"

        try:
            symlink_dir.symlink_to(target_dir)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        import logging

        with caplog.at_level(logging.WARNING):
            validator = PathValidator()
            validator.validate_directory(str(symlink_dir), allow_symlinks=True)

        # Check that a security warning was logged
        assert any(
            "symlink" in record.message.lower() or "security" in record.message.lower()
            for record in caplog.records
        )


class TestAdditionalCoverage:
    """Additional tests to ensure high code coverage."""

    def test_validate_directory_oserror_during_resolve(self):
        """Test that OSError during path resolution is handled."""
        validator = PathValidator()

        # A path with null byte should raise an error during resolution
        with pytest.raises((ValueError, TypeError, OSError)):
            validator.validate_directory("/path/with\x00null")

    def test_validate_file_empty_path(self):
        """Test that empty path in validate_file raises ValueError."""
        validator = PathValidator()

        with pytest.raises(ValueError, match="empty"):
            validator.validate_file("", allowed_extensions=[".sql"])

    def test_validate_file_none_path(self):
        """Test that None path in validate_file raises TypeError."""
        validator = PathValidator()

        with pytest.raises(TypeError, match="None"):
            validator.validate_file(None, allowed_extensions=[".sql"])

    def test_validate_file_oserror_during_resolve(self):
        """Test that OSError during file path resolution is handled."""
        validator = PathValidator()

        with pytest.raises((ValueError, TypeError, OSError)):
            validator.validate_file("/path/with\x00null.sql", allowed_extensions=[".sql"])

    def test_fullwidth_period_traversal_detected(self, tmp_path):
        """Test that fullwidth period traversal sequences are detected."""
        validator = PathValidator()

        # Fullwidth period: U+FF0E
        fullwidth_traversal = str(tmp_path) + "/\uff0e\uff0e/etc"

        with pytest.raises(ValueError, match="[Tt]raversal"):
            validator.validate_directory(fullwidth_traversal)

    def test_windows_reserved_name_empty_string(self):
        """Test that empty string is not considered a reserved name."""
        validator = PathValidator()

        assert validator._is_windows_reserved_name("") is False

    def test_pattern_without_extension(self):
        """Test pattern that has no extractable extension."""
        validator = PathValidator()

        # Pattern with only wildcard, no extension
        result = validator.validate_glob_pattern("subdir/*", allowed_extensions=[".sql"])
        assert result == "subdir/*"

    def test_pattern_starting_with_dot(self):
        """Test pattern for hidden files (starting with dot)."""
        validator = PathValidator()

        # Hidden file pattern - should not extract extension incorrectly
        result = validator.validate_glob_pattern(".hidden", allowed_extensions=[".sql"])
        assert result == ".hidden"

    def test_safe_read_oserror_during_resolve(self, tmp_path):
        """Test that OSError during resolution in safe read is handled."""
        test_file = tmp_path / "test.sql"
        test_file.write_text("SELECT 1")

        # Create a custom mock that only fails on the second resolve call
        resolve_count = [0]
        original_resolve = Path.resolve

        def mock_resolve(self):
            resolve_count[0] += 1
            if resolve_count[0] == 1:
                # First call is for base_dir - let it work
                return original_resolve(self)
            else:
                # Second call is for path - make it fail
                raise OSError("Mocked error")

        with patch.object(Path, "resolve", mock_resolve):
            with pytest.raises(ValueError, match="Invalid path"):
                _safe_read_sql_file(test_file, base_dir=tmp_path)

    def test_safe_read_generic_oserror(self, tmp_path):
        """Test that generic OSError during read is handled."""
        test_file = tmp_path / "test.sql"
        test_file.write_text("SELECT 1")

        # Mock read_text to raise a generic OSError (not FileNotFoundError or PermissionError)
        with patch.object(Path, "read_text", side_effect=OSError("Disk error")):
            with pytest.raises(ValueError, match="Cannot read SQL file"):
                _safe_read_sql_file(test_file, base_dir=tmp_path)

    def test_validate_file_with_windows_reserved_name(self, tmp_path):
        """Test that Windows reserved names trigger error in validate_file."""
        validator = PathValidator()

        # Create a file with reserved name stem (on non-Windows, file can exist)
        reserved_file = tmp_path / "CON.sql"
        reserved_file.write_text("SELECT 1")

        with pytest.raises(ValueError, match="[Rr]eserved"):
            validator.validate_file(str(reserved_file), allowed_extensions=[".sql"])

    def test_validate_file_symlink_warning_logged(self, tmp_path, caplog):
        """Test that symlink warning is logged for files too."""
        target_file = tmp_path / "target.sql"
        target_file.write_text("SELECT 1")
        symlink_file = tmp_path / "link.sql"

        try:
            symlink_file.symlink_to(target_file)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        import logging

        with caplog.at_level(logging.WARNING):
            validator = PathValidator()
            validator.validate_file(
                str(symlink_file),
                allowed_extensions=[".sql"],
                allow_symlinks=True,
            )

        # Check that a security warning was logged
        assert any(
            "symlink" in record.message.lower() or "security" in record.message.lower()
            for record in caplog.records
        )

    def test_glob_pattern_double_star_all_files(self):
        """Test that **/* pattern is accepted like * pattern."""
        validator = PathValidator()

        result = validator.validate_glob_pattern("**/*", allowed_extensions=[".sql"])
        assert result == "**/*"

    def test_whitespace_only_path_rejected(self):
        """Test that whitespace-only path is rejected."""
        validator = PathValidator()

        with pytest.raises(ValueError, match="empty"):
            validator.validate_directory("   ")

    def test_whitespace_only_pattern_rejected(self):
        """Test that whitespace-only pattern is rejected."""
        validator = PathValidator()

        with pytest.raises(ValueError, match="[Ee]mpty"):
            validator.validate_glob_pattern("   ", allowed_extensions=[".sql"])

    def test_safe_read_file_deleted_after_exists_check(self, tmp_path):
        """Test that FileNotFoundError during read is handled (TOCTOU race)."""
        test_file = tmp_path / "test.sql"
        test_file.write_text("SELECT 1")

        # Mock the exists check to return True, but read_text to raise FileNotFoundError
        # This simulates the file being deleted between the check and read
        def mock_read_text(self, *args, **kwargs):
            raise FileNotFoundError("File was deleted")

        with patch.object(Path, "read_text", mock_read_text):
            with pytest.raises(FileNotFoundError, match="not found"):
                _safe_read_sql_file(test_file, base_dir=tmp_path)

    def test_fullwidth_period_traversal_only(self):
        """Test that isolated fullwidth period traversal is detected."""
        validator = PathValidator()

        # Create a path with only fullwidth periods
        # Note: NFKC normalization converts fullwidth periods to regular periods
        # So this tests the post-normalization check
        path_with_fullwidth = "/tmp/\uff0e\uff0e/etc"

        # This should be caught either by normalization or by the check
        with pytest.raises((ValueError, FileNotFoundError)):
            validator.validate_directory(path_with_fullwidth)

    def test_validate_directory_oserror_with_specific_mock(self, tmp_path):
        """Test OSError during resolve in validate_directory."""
        validator = PathValidator()

        # Create a directory path
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()

        # Use a more targeted mock
        original_resolve = Path.resolve

        def mock_resolve(self):
            if "valid" in str(self):
                raise OSError("Cannot resolve path")
            return original_resolve(self)

        with patch.object(Path, "resolve", mock_resolve):
            with pytest.raises(ValueError, match="Invalid path"):
                validator.validate_directory(str(valid_dir))

    def test_validate_file_oserror_with_specific_mock(self, tmp_path):
        """Test OSError during resolve in validate_file."""
        validator = PathValidator()

        # Create a file
        test_file = tmp_path / "test.sql"
        test_file.write_text("SELECT 1")

        original_resolve = Path.resolve

        def mock_resolve(self):
            if "test.sql" in str(self):
                raise OSError("Cannot resolve path")
            return original_resolve(self)

        with patch.object(Path, "resolve", mock_resolve):
            with pytest.raises(ValueError, match="Invalid path"):
                validator.validate_file(str(test_file), allowed_extensions=[".sql"])
