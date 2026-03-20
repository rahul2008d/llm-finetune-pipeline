"""Tests that validate the monorepo directory layout, file conventions, and config integrity.

Run with::

    pytest tests/test_repo_structure.py -v
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

SRC_SUBDIRS = ["config", "data", "training", "evaluation", "serving", "monitoring", "utils"]


# ===================================================================
# 1. Required directories exist
# ===================================================================

REQUIRED_DIRS = [
    "src/config",
    "src/data",
    "src/training",
    "src/evaluation",
    "src/serving",
    "src/monitoring",
    "src/utils",
    "tests/unit",
    "tests/integration",
    "tests/e2e",
    "configs/training",
    "configs/evaluation",
    "configs/deployment",
    "docker",
    "scripts",
]


@pytest.mark.parametrize("rel_path", REQUIRED_DIRS)
def test_required_directory_exists(rel_path: str) -> None:
    """Each required directory must be present in the repo."""
    assert (ROOT / rel_path).is_dir(), f"Directory missing: {rel_path}"


# ===================================================================
# 2. Required files exist
# ===================================================================

REQUIRED_FILES = [
    "pyproject.toml",
    "Makefile",
    ".pre-commit-config.yaml",
    ".env.example",
    ".gitignore",
    "docker/Dockerfile.train",
    "docker/Dockerfile.serve",
]


@pytest.mark.parametrize("rel_path", REQUIRED_FILES)
def test_required_file_exists(rel_path: str) -> None:
    """Each required top-level / docker file must be present."""
    assert (ROOT / rel_path).is_file(), f"File missing: {rel_path}"


# ===================================================================
# 3. src/ sub-package conventions
# ===================================================================


@pytest.mark.parametrize("pkg", SRC_SUBDIRS)
def test_src_package_has_init(pkg: str) -> None:
    """Every src/ subdirectory must contain an __init__.py."""
    assert (ROOT / "src" / pkg / "__init__.py").is_file(), f"src/{pkg}/__init__.py missing"


@pytest.mark.parametrize("pkg", SRC_SUBDIRS)
def test_src_package_has_py_typed(pkg: str) -> None:
    """Every src/ subdirectory must contain a py.typed marker file."""
    assert (ROOT / "src" / pkg / "py.typed").is_file(), f"src/{pkg}/py.typed missing"


@pytest.mark.parametrize("pkg", SRC_SUBDIRS)
def test_src_package_init_has_all(pkg: str) -> None:
    """Every __init__.py must define __all__ = [...]."""
    init_path = ROOT / "src" / pkg / "__init__.py"
    source = init_path.read_text()
    tree = ast.parse(source, filename=str(init_path))
    all_found = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            (isinstance(t, ast.Name) and t.id == "__all__")
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        for node in ast.walk(tree)
    )
    assert all_found, f"src/{pkg}/__init__.py does not define __all__"


@pytest.mark.parametrize("pkg", SRC_SUBDIRS)
def test_src_package_init_has_module_docstring(pkg: str) -> None:
    """Every __init__.py must start with a module-level docstring."""
    init_path = ROOT / "src" / pkg / "__init__.py"
    source = init_path.read_text()
    tree = ast.parse(source, filename=str(init_path))
    docstring = ast.get_docstring(tree)
    assert docstring, f"src/{pkg}/__init__.py has no module docstring"


# ===================================================================
# 4. Every .py file in src/ has a docstring and uses absolute imports
# ===================================================================

_SRC_PY_FILES = sorted(
    p
    for p in (ROOT / "src").rglob("*.py")
    if p.is_file()
)


@pytest.mark.parametrize("py_file", _SRC_PY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_src_py_file_has_module_docstring(py_file: Path) -> None:
    """Every .py file under src/ must have a module-level docstring."""
    source = py_file.read_text()
    tree = ast.parse(source, filename=str(py_file))
    docstring = ast.get_docstring(tree)
    assert docstring, f"{py_file.relative_to(ROOT)} has no module docstring"


_RELATIVE_IMPORT_RE = re.compile(r"^\s*from\s+\.\s*\S+\s+import", re.MULTILINE)


@pytest.mark.parametrize("py_file", _SRC_PY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_src_py_file_no_relative_imports(py_file: Path) -> None:
    """No .py file under src/ should use relative imports ('from . import ...')."""
    source = py_file.read_text()
    matches = _RELATIVE_IMPORT_RE.findall(source)
    assert not matches, (
        f"{py_file.relative_to(ROOT)} contains relative import(s): {matches}"
    )


# ===================================================================
# 5. pyproject.toml integrity
# ===================================================================


def _read_pyproject() -> str:
    return (ROOT / "pyproject.toml").read_text()


def test_pyproject_has_project_scripts_with_llm_ft() -> None:
    """pyproject.toml must declare [project.scripts] with 'llm-ft' entry."""
    content = _read_pyproject()
    assert "[project.scripts]" in content, "[project.scripts] section missing"
    assert "llm-ft" in content, "'llm-ft' entry missing in [project.scripts]"


def test_pyproject_has_ruff_section() -> None:
    """pyproject.toml must contain a [tool.ruff] section."""
    content = _read_pyproject()
    assert "[tool.ruff]" in content, "[tool.ruff] section missing"


def test_pyproject_has_mypy_strict() -> None:
    """pyproject.toml must contain [tool.mypy] with strict = true."""
    content = _read_pyproject()
    assert "[tool.mypy]" in content, "[tool.mypy] section missing"
    assert re.search(r"strict\s*=\s*true", content), "mypy strict=true not found"


def test_pyproject_has_pytest_ini_options() -> None:
    """pyproject.toml must contain [tool.pytest.ini_options]."""
    content = _read_pyproject()
    assert "[tool.pytest.ini_options]" in content, "[tool.pytest.ini_options] section missing"


# ===================================================================
# 6. .gitignore entries
# ===================================================================

EXPECTED_GITIGNORE_PATTERNS = [
    ".env",
    "__pycache__",
    "*.pyc",
    ".terraform",
    "*.safetensors",
    "*.bin",
    "wandb/",
    "data/",
]


def _read_gitignore() -> str:
    return (ROOT / ".gitignore").read_text()


@pytest.mark.parametrize("pattern", EXPECTED_GITIGNORE_PATTERNS)
def test_gitignore_contains_pattern(pattern: str) -> None:
    """The .gitignore must contain each expected pattern on its own line."""
    content = _read_gitignore()
    # Check that the pattern appears as a line (possibly with trailing whitespace)
    escaped = re.escape(pattern)
    assert re.search(rf"^\s*{escaped}\s*$", content, re.MULTILINE), (
        f".gitignore missing pattern: {pattern}"
    )
