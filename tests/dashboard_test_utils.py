from pathlib import Path


DEMO_ENGINE = (
    "from causal_edge.engine.base import StrategyEngine\n"
    "class DemoSignalEngine(StrategyEngine):\n"
    "    def compute_signals(self):\n"
    "        raise NotImplementedError\n"
    "    def get_latest_signal(self):\n"
    "        return {'position': 0.0}\n"
)


def write_demo_project(
    *, paper_log=False, cta=False, backtest_csv=None, paper_csv=None, settings_lines=None
):
    config = list(settings_lines or ["settings: {}"])
    config.extend(
        [
        "strategies:",
        "  - id: demo_signal",
        '    name: "Demo Signal"',
        "    asset: ETHUSD",
        '    color: "#2563EB"',
        "    engine: strategies.demo_signal.engine",
        "    trade_log: data/trade_log_demo_signal.csv",
        ]
    )
    if paper_log:
        config.append("    paper_log: data/paper_log_demo_signal.csv")
    config.append('    thesis: "Signal thesis"')
    if cta:
        config.append('    cta_text: "Start tracking this signal"')

    Path("strategies.yaml").write_text("\n".join(config) + "\n", encoding="utf-8")
    Path("strategies").mkdir()
    Path("strategies/__init__.py").write_text("", encoding="utf-8")
    Path("strategies/demo_signal").mkdir(parents=True)
    Path("strategies/demo_signal/__init__.py").write_text("", encoding="utf-8")
    Path("strategies/demo_signal/engine.py").write_text(DEMO_ENGINE, encoding="utf-8")
    Path("data").mkdir()
    if backtest_csv is not None:
        Path("data/trade_log_demo_signal.csv").write_text(backtest_csv, encoding="utf-8")
    if paper_csv is not None:
        Path("data/paper_log_demo_signal.csv").write_text(paper_csv, encoding="utf-8")
