"""Structural tests — enforce architecture mechanically.

These tests validate project conventions, not business logic.
Every assertion includes a 'Fix:' instruction.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
YAML_PATH = ROOT / "strategies.yaml"

REQUIRED_FIELDS = ("id", "name", "asset", "color", "engine", "trade_log")
HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

REQUIRED_AGENTS_MD = [
    "causal_edge/engine/AGENTS.md",
    "causal_edge/dashboard/AGENTS.md",
    "causal_edge/validation/AGENTS.md",
    "causal_edge/plugins/AGENTS.md",
    "strategies/AGENTS.md",
]


@pytest.fixture
def strategies():
    with open(YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("strategies") or []


# ── Tests 1-4: YAML schema ──────────────────────────────────────


class TestYamlHasRequiredFields:
    def test_yaml_has_required_fields(self, strategies):
        for strat in strategies:
            for field in REQUIRED_FIELDS:
                assert field in strat, (
                    f"Strategy '{strat.get('id', '???')}' is missing '{field}'.\n"
                    f"Fix: Add '{field}' to the strategy entry in strategies.yaml.\n"
                    f"See: docs/add-strategy.md"
                )


class TestStrategyIdsUnique:
    def test_strategy_ids_unique(self, strategies):
        ids = [s["id"] for s in strategies]
        assert len(ids) == len(set(ids)), (
            f"Duplicate strategy IDs: {ids}.\n"
            f"Fix: Ensure each strategy has a unique 'id' in strategies.yaml."
        )


class TestEngineModuleImportable:
    def test_engine_module_importable(self, strategies):
        for strat in strategies:
            module_path = strat["engine"]
            try:
                importlib.import_module(module_path)
            except ImportError as e:
                pytest.fail(
                    f"Cannot import engine '{module_path}' "
                    f"for strategy '{strat['id']}': {e}\n"
                    f"Fix: Check that the module path in strategies.yaml matches "
                    f"the actual file location, and that __init__.py exists."
                )


class TestColorsAreValidHex:
    def test_colors_are_valid_hex(self, strategies):
        for strat in strategies:
            color = strat["color"]
            assert HEX_COLOR_RE.match(color), (
                f"Strategy '{strat['id']}' has invalid color '{color}'.\n"
                f"Fix: Use 6-digit hex like '#FF2D55'."
            )


# ── Test 5: File size ────────────────────────────────────────────


class TestFileSizeLimit:
    MAX_LINES = 400

    def test_no_file_exceeds_limit(self):
        py_files = [
            f
            for f in ROOT.rglob("*.py")
            if "__pycache__" not in str(f) and ".venv" not in str(f) and "build" not in f.parts
        ]
        violations = []
        for f in py_files:
            lines = len(f.read_text(encoding="utf-8").splitlines())
            if lines > self.MAX_LINES:
                violations.append(f"{f.relative_to(ROOT)}: {lines} lines")
        assert not violations, (
            f"Files exceeding {self.MAX_LINES} lines:\n"
            + "\n".join(f"  {v}" for v in violations)
            + f"\nFix: Split into smaller modules. See ARCHITECTURE.md."
        )


# ── Test 6: AGENTS.md existence ──────────────────────────────────


class TestSubsystemAgentsMd:
    def test_agents_md_exists(self):
        missing = [p for p in REQUIRED_AGENTS_MD if not (ROOT / p).exists()]
        assert not missing, (
            f"Missing AGENTS.md files:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nFix: Create each missing file with 'I want to...' decision tree."
        )


# ── Test 7: Components registered ────────────────────────────────


class TestComponentsRegistered:
    def test_components_used_in_generator(self):
        comp_path = ROOT / "causal_edge/dashboard/components.py"
        gen_path = ROOT / "causal_edge/dashboard/generator.py"
        if not comp_path.exists() or not gen_path.exists():
            pytest.skip("Dashboard not yet implemented")

        comp_src = comp_path.read_text(encoding="utf-8")
        gen_src = gen_path.read_text(encoding="utf-8")

        tree = ast.parse(comp_src)
        public_funcs = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        ]
        unused = [f for f in public_funcs if f not in gen_src]
        assert not unused, (
            f"Components not used in generator.py: {unused}\n"
            f"Fix: Import and register in generator.py env.globals, "
            f"or prefix with _ to mark private."
        )


# ── Test 8: AGENTS.md has decision tree ──────────────────────────


class TestAgentsMdHasDecisionTree:
    def test_all_agents_md_have_decision_tree(self):
        agents_files = list(ROOT.rglob("AGENTS.md"))
        missing = []
        for af in agents_files:
            content = af.read_text(encoding="utf-8")
            if "I want to..." not in content and "## I want to" not in content:
                missing.append(str(af.relative_to(ROOT)))
        assert not missing, (
            f"AGENTS.md files missing 'I want to...' decision tree:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nFix: Add '## I want to...' section with task-based routing."
        )


# ── Test 9: Plugins optional ────────────────────────────────────


class TestPluginsOptional:
    def test_core_imports_without_plugins(self):
        """Core modules must not depend on plugins at import time."""
        core_modules = [
            "causal_edge.config",
            "causal_edge.engine.base",
            "causal_edge.validation.metrics",
            "causal_edge.validation.gate",
        ]
        for mod in core_modules:
            try:
                importlib.import_module(mod)
            except ImportError as e:
                if "plugins" in str(e).lower():
                    pytest.fail(
                        f"Core module '{mod}' depends on plugins: {e}\n"
                        f"Fix: Use try/except ImportError for plugin imports."
                    )

    def test_core_source_no_plugin_imports(self):
        """Core source files must not import from plugins (except cli.py with try/except)."""
        violations = []
        core_dirs = [ROOT / "causal_edge" / d for d in ("engine", "dashboard", "validation")]
        core_files = [ROOT / "causal_edge" / "config.py"]
        for d in core_dirs:
            if d.exists():
                core_files.extend(f for f in d.rglob("*.py") if "__pycache__" not in str(f))
        for f in core_files:
            content = f.read_text(encoding="utf-8")
            if re.search(r"(?:from|import)\s+causal_edge\.plugins", content):
                violations.append(str(f.relative_to(ROOT)))
        assert not violations, (
            f"Core files import from plugins:\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\nFix: Plugins are optional. Core must not import them. "
            + "Use try/except in cli.py only."
        )


# ── Test 10: Strategies standalone ───────────────────────────────


class TestStrategiesStandalone:
    def test_strategies_no_framework_imports(self):
        """strategies/ must not import causal_edge internals (except base.py)."""
        strat_dir = ROOT / "strategies"
        if not strat_dir.exists():
            pytest.skip("No strategies directory")
        violations = []
        for f in strat_dir.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            content = f.read_text(encoding="utf-8")
            imports = re.findall(r"(?:from|import)\s+(causal_edge\.[a-zA-Z_.]+)", content)
            bad = [i for i in imports if i != "causal_edge.engine.base"]
            if bad:
                violations.append(f"{f.relative_to(ROOT)}: imports {bad}")
        assert not violations, (
            f"strategies/ files import causal_edge internals:\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\nFix: strategies/ may only import causal_edge.engine.base. "
            + "Keep engines standalone."
        )


# ── Test 11: CLI entry points ────────────────────────────────────


class TestCliEntryPoints:
    def test_cli_subcommands_exist(self):
        """CLI must have init, run, dashboard, validate, discover, status subcommands."""
        from causal_edge.cli import main

        required = {"init", "run", "dashboard", "validate", "discover", "status"}
        actual = set(main.commands.keys())
        missing = required - actual
        assert not missing, (
            f"CLI missing subcommands: {missing}\n"
            f"Fix: Add @main.command() for each in causal_edge/cli.py."
        )


# ── Test 12: No hardcoded paths ──────────────────────────────────


class TestNoHardcodedPaths:
    def test_no_absolute_paths_in_python(self):
        """No hardcoded absolute paths like /home/... or ~/..."""
        pattern = re.compile(r'["\'](?:/home/|/Users/|~/Claude/)[^"\']*["\']')
        violations = []
        for f in ROOT.rglob("*.py"):
            if "__pycache__" in str(f) or ".venv" in str(f):
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{f.relative_to(ROOT)}:{i}")
        assert not violations, (
            f"Hardcoded absolute paths found:\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\nFix: Use Path(__file__).parent, config paths, "
            + "or environment variables instead."
        )


# ── Test 13: No secrets ─────────────────────────────────────────


class TestNoSecrets:
    def test_no_secrets_in_source(self):
        """No API keys or tokens in source files."""
        patterns = [
            re.compile(r'(?:API_KEY|SECRET|TOKEN)\s*=\s*["\'][a-zA-Z0-9]{10,}["\']'),
            re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        ]
        violations = []
        for f in ROOT.rglob("*.py"):
            if "__pycache__" in str(f) or ".venv" in str(f):
                continue
            content = f.read_text(encoding="utf-8")
            for pat in patterns:
                if pat.search(content):
                    violations.append(str(f.relative_to(ROOT)))
                    break
        assert not violations, (
            f"Possible secrets in source:\n"
            + "\n".join(f"  {v}" for v in violations)
            + "\nFix: Use environment variables or .env files (gitignored)."
        )


# ── Test 14: AGENTS.md references exist ──────────────────────────


class TestAgentsMdReferencesExist:
    def test_all_referenced_files_exist(self):
        """Every file path in backticks in AGENTS.md must exist."""
        agents_files = list(ROOT.rglob("AGENTS.md"))
        missing = []
        for af in agents_files:
            content = af.read_text(encoding="utf-8")
            paths = re.findall(r"`([a-zA-Z_][a-zA-Z0-9_./]*\.[a-z]{1,4})`", content)
            for p in paths:
                # Try relative to AGENTS.md parent, then relative to ROOT
                if not (af.parent / p).exists() and not (ROOT / p).exists():
                    missing.append(f"{af.relative_to(ROOT)}: references `{p}`")
        assert not missing, (
            f"AGENTS.md files reference non-existent paths:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nFix: Update the path or create the missing file."
        )


# ── Test 15: AGENTS.md size budget ───────────────────────────────


class TestAgentsMdSizeBudget:
    def test_root_agents_md_under_80_lines(self):
        root_md = ROOT / "AGENTS.md"
        if not root_md.exists():
            pytest.skip("No root AGENTS.md")
        n = len(root_md.read_text(encoding="utf-8").splitlines())
        assert n <= 80, (
            f"Root AGENTS.md is {n} lines (max 80).\n"
            f"Fix: Move details to subsystem AGENTS.md files."
        )

    def test_subsystem_agents_md_under_60_lines(self):
        for af in ROOT.rglob("AGENTS.md"):
            if af == ROOT / "AGENTS.md":
                continue
            n = len(af.read_text(encoding="utf-8").splitlines())
            assert n <= 60, (
                f"{af.relative_to(ROOT)} is {n} lines (max 60).\n"
                f"Fix: Split into sub-sections or link to docs/."
            )
