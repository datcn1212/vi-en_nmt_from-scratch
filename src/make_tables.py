"""Generate the report's results table straight from results/*.json
"""
import glob
import json
import os

OUT_MD = "results/bleu_table.md"
OUT_TEX = "report/tables/bleu.tex"


def load_test_results():
    rows = []
    for path in glob.glob("results/*.json"):
        d = json.load(open(path))
        if d.get("split") == "test":
            rows.append(d)
    rows.sort(key=lambda d: d["bleu"], reverse=True)
    return rows


def write_markdown(rows):
    lines = ["| system | decode | BLEU | n |", "|---|---|---|---|"]
    for d in rows:
        lines.append(f"| {d['system']} | {d['decode']} | {d['bleu']:.2f} | {d['n_sentences']} |")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def write_latex(rows):
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"system & decode & BLEU \\",
        r"\midrule",
    ]
    for d in rows:
        system = d["system"].replace("_", r"\_")
        lines.append(f"{system} & {d['decode']} & {d['bleu']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_TEX}")


def main():
    rows = load_test_results()
    write_markdown(rows)
    write_latex(rows)


if __name__ == "__main__":
    main()
