#!/usr/bin/env python3
"""Build one offline HTML file using only Python's standard library."""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(output: Path):
    data = json.loads((HERE / "data.json").read_text())
    assert len(data["experiments"]) == data["counts"]["experiments"] == 117
    assert len({e["id"] for e in data["experiments"]}) == 117
    template = (HERE / "template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # Prevent embedded research text from closing the non-executable data block.
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = template.replace("__STYLE__", (HERE / "style.css").read_text()).replace("__DATA__", payload).replace("__APP__", (HERE / "app.js").read_text())
    assert not re.search(r"__(?:STYLE|DATA|APP)__", html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"{output}: {output.stat().st_size:,} bytes, {len(data['experiments'])} experiments, no runtime dependencies")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE / "index.html")
    build(p.parse_args().output)
