"""Print the manuscript's reported aggregates without running an evaluation."""
from pathlib import Path
import json


def main():
    path = Path(__file__).resolve().parents[1] / "paper" / "results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    print("HMBR | paper-reported WF1 (%)")
    for row in data["main_results"]:
        values = ", ".join(f"{metric.upper()} {value:.2f}%" for metric, value in row["scores"].items())
        print(f"{row['dataset']}: {values} | {row['statistic']}")
    print("\nM3ED module ablations (Table 4, WF1 percentage points)")
    for row in data["mechanism_results"]:
        if row["component"] == "whole module":
            print(f"{row['module']}: +{row['gain']:.3f} pp")


if __name__ == "__main__":
    main()
