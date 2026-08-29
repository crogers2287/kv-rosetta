"""Every module and script must not reference a name that cannot exist.

This exists because a cross-backend run completed both transfers, spent ten minutes of GPU
time doing it, and then died assembling the record it was about to write: a refactor had
replaced a local `revisions` dict with a function call, and one later reference to it
survived. ast.parse accepts that happily - it is valid syntax - and the line only executes
at the very end, after everything expensive.

The check is deliberately conservative. It reports a name only when it is not a builtin, not
bound anywhere in its function, not bound at module level, and not a parameter, so a false
positive means a real ambiguity rather than a style opinion. Missing some real faults is
acceptable; failing on working code is not, because a check that cries wolf gets skipped.
"""

import ast
import builtins
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__",
                                 "__package__", "__builtins__", "__class__"}


def _bound_names(node) -> set[str]:
    """Every name a scope binds: assignment, import, def, class, with, for, except, match,
    walrus. Some of these carry the bound name as a plain string rather than a Name node,
    which is how `except ValueError as exc` slipped through the first version."""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.arg):
            names.add(child.arg)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            names.update(child.names)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)          # `except E as exc` binds a str, not a Name node
        elif isinstance(child, ast.MatchAs) and child.name:
            names.add(child.name)
        elif isinstance(child, ast.MatchStar) and child.name:
            names.add(child.name)
    return names


def undefined_names(source: str) -> list[str]:
    tree = ast.parse(source)
    module_level = _bound_names(tree)
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = _bound_names(node) | module_level | BUILTINS
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id not in local:
                    problems.append(f"{node.name}: {child.id} (line {child.lineno})")
    return problems


class UndefinedNameTest(unittest.TestCase):
    def files(self):
        for directory in ("kv_rosetta", "scripts", "tests"):
            yield from sorted((ROOT / directory).rglob("*.py"))

    def test_no_module_references_a_name_it_cannot_resolve(self):
        for path in self.files():
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertEqual(undefined_names(path.read_text()), [])

    def test_the_check_finds_the_fault_it_was_written_for(self):
        """The exact shape that survived a refactor and killed a completed run."""
        self.assertIn("main: revisions (line 3)", undefined_names(
            "def main():\n"
            "    ok = check({'a': 1})\n"
            "    return next(iter(revisions.values()))\n"))

    def test_the_check_does_not_fire_on_ordinary_code(self):
        for source in ("def f(a, b=2):\n    c = a + b\n    return c\n",
                       "import os\ndef f():\n    return os.getcwd()\n",
                       "X = 1\ndef f():\n    return X\n",
                       "def f(items):\n    return [y for y in items if y]\n",
                       "def f():\n    with open('x') as h:\n        return h.read()\n",
                       "def f():\n    try:\n        pass\n"
                       "    except ValueError as exc:\n        return exc\n",
                       "def f(v):\n    if (n := len(v)):\n        return n\n    return 0\n"):
            with self.subTest(source=source.splitlines()[0]):
                self.assertEqual(undefined_names(source), [])


if __name__ == "__main__":
    unittest.main()
