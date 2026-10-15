import json

from chuckcheck.runner import run_demo


def test_demo_writes_all_outputs(tmp_path):
    res, paths, b = run_demo(tmp_path, extension_deg=18, fps=60)
    for k in ("video", "json", "csv", "html"):
        assert (tmp_path / paths[k].split("/")[-1]).exists() or paths[k]
    d = json.loads((tmp_path / "analysis.json").read_text())
    assert d["deliveries"][0]["legality"]["verdict"] in {"LEGAL", "LIKELY LEGAL", "BORDERLINE", "SUSPECT"}
    html = (tmp_path / "report.html").read_text()
    assert "CHUCKCHECK" in html and "plotly" in html.lower()
    assert (tmp_path / "annotated.mp4").stat().st_size > 10_000
