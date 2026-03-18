"""Quick log replay utility for KPI and failure trend inspection.

Usage:
  uv run python baseline/replay_metrics.py --log baseline/agent.log
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

KPI_RE = re.compile(
    r"KPI \[(?P<label>current|previous) turn (?P<turn>\d+)\].*clients=(?P<clients>\d+).*served=(?P<served>\d+).*revenue=(?P<revenue>-?\d+).*bid_spend=(?P<bid_spend>-?\d+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="agent.log", help="Path to runtime log file")
    parser.add_argument("--last", type=int, default=10, help="How many KPI lines to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kpis: list[dict] = []
    failures = Counter()

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = KPI_RE.search(line)
            if m:
                kpis.append(
                    {
                        "label": m.group("label"),
                        "turn": int(m.group("turn")),
                        "clients": int(m.group("clients")),
                        "served": int(m.group("served")),
                        "revenue": int(m.group("revenue")),
                        "bid_spend": int(m.group("bid_spend")),
                    }
                )
            low = line.lower()
            if "dish not found in kitchen or not ready" in low:
                failures["not_ready"] += 1
            if "client is not waiting in your restaurant" in low:
                failures["not_waiting"] += 1
            if "meals fetch failed" in low and "turn_id=0" in low:
                failures["turn_id_zero_meals"] += 1

    if not kpis:
        print("No KPI lines found.")
        return

    tail = kpis[-max(1, args.last):]
    print("Last KPI entries:")
    for row in tail:
        clients = row["clients"]
        served = row["served"]
        conv = (served / clients * 100.0) if clients > 0 else 0.0
        net = row["revenue"] - row["bid_spend"]
        print(
            f"turn={row['turn']:>3} [{row['label']}] clients={clients:>3} served={served:>3} "
            f"conv={conv:>5.1f}% revenue={row['revenue']:>5} bid_spend={row['bid_spend']:>5} net={net:>5}"
        )

    print("\nFailure counters:")
    for key, value in sorted(failures.items()):
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
