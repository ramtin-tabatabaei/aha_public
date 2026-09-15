"""Compatibility facade for the split BT maker modules.

The implementation now lives in focused sibling modules. This file keeps
``from aha_publish.behavior_trees.logic import main`` and older direct imports working.
"""

from aha_publish import paths

from importlib import import_module

if __package__:
    _PACKAGE = __package__
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(paths.SOURCE_DIR))
    _PACKAGE = "aha_publish.behavior_trees"

_MODULES = (
    "config",
    "prompts",
    "io_context",
    "failure_definitions",
    "conditions",
    "semantics",
    "cleanup",
    "prompt_building",
    "local_generation",
    "llm_generation",
    "gui",
    "cli",
)

for _module_name in _MODULES:
    _module = import_module(f"{_PACKAGE}.{_module_name}")
    globals().update(
        {
            _name: _value
            for _name, _value in vars(_module).items()
            if not _name.startswith("__")
        }
    )

del import_module, _module, _module_name, _MODULES, _PACKAGE
