import importlib.util
from pathlib import Path


def test_package_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "zpackage" / "__init__.py").exists()
    assert (root / "src" / "zpackage" / "wmt_refactored.py").exists()
    assert (root / "src" / "zpackage" / "ztake_refactored.py").exists()


def test_refactored_modules_parse_without_importing_dependencies():
    root = Path(__file__).resolve().parents[1]
    for relpath in ["src/zpackage/wmt_refactored.py", "src/zpackage/ztake_refactored.py"]:
        path = root / relpath
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None
