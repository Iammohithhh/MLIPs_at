import pytest

from atlas_pipeline.cleavage import data_class, parse

ROW = "<tr>" + "".join("<td>{}</td>" for _ in range(12)) + "</tr>"


def page(t2_rows: list[list[str]]) -> str:
    t1 = (
        '<figure><figcaption>Table 1: The 19 state-of-the-art uMLIPs considered in this study</figcaption><table>'
        "<tr><th>Model Family</th><th>Model Name</th><th>#Params</th><th>Training Data Description</th></tr>"
        "<tr><td>MACE</td><td>MACE-OMAT</td><td>9.06M</td><td>OMat24: Non-equilibrium focus</td></tr>"
        "<tr><td></td><td>MACE-MP-0b3</td><td>4.69M</td><td>Materials Project: Equilibrium only</td></tr>"
        "</table></figure>"
    )
    t2 = '<figure><figcaption>Table 2: Comprehensive performance evaluation of 19 uMLIPs</figcaption><table>'
    t2 += "<tr><td>Model</td><td>APE Mean</td></tr><tr><td>MACE</td></tr>"
    t2 += "".join(ROW.format(*r) for r in t2_rows) + "</table></figure>"
    return t1 + t2


GOOD = [
    ["Mean Baseline", "48.7", "28.6", "33.8", "46", "0.000", "2816", "—", "—", "—", "1", "—"],
    ["MACE-OMAT", "7.8", "6.1", "6.9", "10", "0.955", "15", "81.2", "3.9", "14.9", "9.06M", "20"],
    ["MACE-MP-0b3", "25.2", "24.6", "24.7", "30", "0.584", "17", "69.9", "4.5", "25.6", "4.69M", "21"],
]


def test_parse_copies_numbers_and_joins_tables():
    d = parse(page(GOOD))
    assert d["source"]["arxiv"] == "2508.21663v1"
    assert d["baseline"]["ape_median"] == 28.6 and d["baseline"]["term"] is None
    a, b = d["models"]
    assert (a["name"], a["family"], a["data_class"], a["ape_median"], a["params"]) == ("MACE-OMAT", "MACE", "omat", 6.1, "9.06M")
    assert (b["family"], b["data_class"], b["r2"], b["term"]) == ("MACE", "mp", 0.584, 69.9)


def test_parse_rejects_unmatched_models():
    with pytest.raises(ValueError):
        parse(page(GOOD[:2]))
    with pytest.raises(ValueError):
        parse(page(GOOD + [["Unknown", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1M", "1"]]))


def test_data_class():
    assert data_class("Combined: OMat24 + MPtrj + sAlex") == "omat-combined"
    assert data_class("OC20: Surface-adsorbate interactions") == "oc20"
    assert data_class("MP + Alexandria (subset)") == "mp-alexandria"
    assert data_class("Proprietary workflow-generated data") == "other"
