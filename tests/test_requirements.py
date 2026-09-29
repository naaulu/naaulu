"""Guard against dependency drift: what the code imports must be declared.

The AUS/USA providers were caught twice by this class of bug (a missing
`s3fs` behind fsspec, an undeclared `geovista`), so it is checked here.
"""

import ast
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "naaulu"

# import name -> distribution name, when they differ
ALIASES = {"PIL": "pillow", "yaml": "pyyaml"}

# stdlib module -> the data package it needs when the OS provides none.
# zoneinfo reads the IANA timezone database: on a bare container the stdlib
# falls back to the `tzdata` package, so an undeclared tzdata turns into
# ZoneInfoNotFoundError at import time.
DATA_PACKAGES = {"zoneinfo": "tzdata"}

# packages pulled in through a string `engine=` rather than an import
ENGINES = {"h5netcdf"}


def _project():
    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]


def _declared():
    """Every distribution named in pyproject, extras included."""
    names = {dep.split("[")[0].split()[0] for dep in _project()["dependencies"]}
    for group in _project().get("optional-dependencies", {}).values():
        names.update(dep.split("[")[0].split()[0] for dep in group)
    return names


def _imported_modules():
    """Every top-level module name the package imports, stdlib included."""
    found = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def _third_party_imports():
    return _imported_modules() - set(sys.stdlib_module_names) - {"naaulu"}


def test_every_import_is_declared():
    declared = _declared()
    undeclared = sorted(
        module for module in _third_party_imports()
        if ALIASES.get(module, module) not in declared
    )
    assert not undeclared, f"imported but not declared in pyproject.toml: {undeclared}"


def test_stdlib_data_packages_are_declared():
    """A stdlib module can still need an undeclared data package.

    zoneinfo ships no timezone database of its own: without a system one it
    falls back to the `tzdata` distribution, so skipping it breaks import
    on any host that lacks /usr/share/zoneinfo - a bare container, say.
    """
    imported = _imported_modules()
    missing = sorted(
        data
        for module, data in DATA_PACKAGES.items()
        if module in imported and data not in _declared()
    )
    assert not missing, (
        f"stdlib module needing a data package not declared in pyproject.toml: {missing}"
    )


def test_string_engines_are_declared():
    missing = sorted(ENGINES - _declared())
    assert not missing, f"used as engine= but not declared in pyproject.toml: {missing}"


def test_plot_dependencies_stay_optional():
    """geovista/pyvista need an OpenGL stack: they belong in the viz extra."""
    hard = {dep.split("[")[0].split()[0] for dep in _project()["dependencies"]}
    assert {"geovista", "pyvista"}.isdisjoint(hard)
    viz = _project()["optional-dependencies"]["viz"]
    assert any(dep.startswith("geovista") for dep in viz)
