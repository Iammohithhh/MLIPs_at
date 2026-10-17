import pytest

from atlas_pipeline.charge_transfer import parse

HEAD = '<figure><table><tr><th></th><th></th><th>3G-HDNNP</th><th>4G-HDNNP</th><th>CACE-LR</th></tr>'
FOOT = '</table><figcaption>Table 1: Test root mean squared errors (RMSE) are reported</figcaption></figure>'


def rows(cells: list[list[str]]) -> str:
    return "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cells)


GOOD = [
    ["", "r cut", "4.23 Å", "4.23 Å", "4.23 Å"],
    ["C <sub>10</sub> H 2 /C 10 H + 3", "E", "2.045", "1.194", "0.73"],
    ["", "F", "231.0", "78.00", "36.9"],
    ["", "r cut", "5.29 Å", "5.29 Å", "5.29 Å"],
    ["Ag + / − 3", "E", "320.2", "1.323", "0.162"],
    ["", "F", "1913", "31.69", "29.0"],
    ["", "r cut", "5.29 Å", "5.29 Å", "5.29 Å"],
    ["Na 8/9 Cl + 8", "E", "2.042", "0.481", "0.21"],
    ["", "F", "76.67", "32.78", "9.78"],
    ["", "r cut", "-", "4.23 Å", "5.5 Å"],
    ["Au 2 -MgO(001)", "E", "-", "0.219", "0.073"],
    ["", "F", "-", "66.00", "7.91"],
]


def test_parse_copies_numbers_with_their_cutoffs():
    d = parse(HEAD + rows(GOOD) + FOOT)
    assert d["source"]["arxiv"] == "2412.15455v1"
    assert d["models"] == ["3G-HDNNP", "4G-HDNNP", "CACE-LR"]
    c10, ag, nacl, au = d["systems"]
    assert c10["key"] == "c10h2" and c10["models"]["4G-HDNNP"] == {"rcut": 4.23, "rcut_printed": "4.23 Å", "E": 1.194, "E_printed": "1.194", "F": 78.0, "F_printed": "78.00"}
    assert ag["models"]["3G-HDNNP"]["F"] == 1913.0
    assert nacl["models"]["CACE-LR"]["E"] == 0.21
    assert au["models"]["3G-HDNNP"] == {"rcut": None, "rcut_printed": "-", "E": None, "E_printed": "-", "F": None, "F_printed": "-"}
    assert au["models"]["CACE-LR"]["rcut"] == 5.5


def test_parse_rejects_unknown_or_missing_systems():
    bad = [r[:] for r in GOOD]
    bad[1][0] = "Something else"
    with pytest.raises(ValueError):
        parse(HEAD + rows(bad) + FOOT)
    with pytest.raises(ValueError):
        parse(HEAD + rows(GOOD[:9]) + FOOT)
    with pytest.raises(ValueError):
        parse("<html></html>")
