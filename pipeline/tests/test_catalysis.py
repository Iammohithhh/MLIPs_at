import pytest

from atlas_pipeline.catalysis import grid, parse

HEAD = "<tr><th>Model family</th><th></th><th>Variant</th><th></th><th>Training data set</th><th>RMSE (eV)</th><th>MaxAE (eV)</th></tr>"
BODY = (
    '<tr><td rowspan="2">Allegro</td><td></td><td rowspan="2">L-0.1</td><td rowspan="2">{</td><td>MPTrj</td><td>0.774 0.774</td><td>4.13</td></tr>'
    "<tr><td></td><td>OMat24 → MPTrj + sAlex</td><td>0.904</td><td>4.62</td></tr>"
    "<tr><td>CHGNet</td><td></td><td></td><td></td><td>MPTrj</td><td>0.927</td><td>3.53</td></tr>"
    '<tr><td colspan="7">SISSO [ 178 ]</td></tr>'
)


def table(n: int, what: str, extra: str = "") -> str:
    return f'<figure class="ltx_table"><figcaption>Table S{n}: Prediction performance of the MLIPs for the {what}.</figcaption><table>{HEAD}{BODY}{extra}</table></figure>'


def test_grid_expands_rowspans():
    g = grid(f"<table>{HEAD}{BODY}</table>")
    assert g[1][:5] == ["Allegro", "", "L-0.1", "{", "MPTrj"]
    assert g[2][0] == "Allegro" and g[2][2] == "L-0.1" and g[2][4] == "OMat24 → MPTrj + sAlex"


def test_parse_reads_every_task_and_keeps_numbers():
    page = "".join(table(n, f"task {n}") for n in range(2, 16))
    d = parse(page)
    assert len(d["tasks"]) == 14 and d["tasks"][0]["table"] == "Table S2"
    rm = d["tasks"][0]["rmse"]
    assert rm["Allegro | L-0.1 | MPTrj"] == 0.774
    assert rm["Allegro | L-0.1 | OMat24 → MPTrj + sAlex"] == 0.904
    assert rm["CHGNet |  | MPTrj"] == 0.927
    kinds = {m["key"]: m["kind"] for m in d["models"]}
    assert kinds["CHGNet |  | MPTrj"] == "mlip"


def test_parse_requires_all_fourteen_tables():
    with pytest.raises(ValueError):
        parse(table(2, "task"))
