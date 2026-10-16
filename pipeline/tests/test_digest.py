from atlas_pipeline.digest import SEP, page_url, parse_log


def test_page_url_maps_files_to_site_paths():
    assert page_url("index") == "/"
    assert page_url("failures/index") == "/failures/"
    assert page_url("applications/batteries") == "/applications/batteries/"
    assert page_url("failures/[id]") is None
    assert page_url("404") is None


def test_parse_log_groups_additions_by_commit():
    text = (
        f"{SEP}abcdef1234567\t2026-10-16T10:00:00+05:30\tAdd batteries\n\n"
        "atlas/content/claims/x-y.yaml\natlas/content/papers/p1.yaml\natlas/src/pages/applications/batteries.astro\nREADME.md\n"
        f"{SEP}0123456789ab\t2026-10-15T10:00:00+05:30\tTooling only\n\npipeline/x.py\n"
    )
    e = parse_log(text)
    assert len(e) == 1
    assert e[0]["date"] == "2026-10-16" and e[0]["commit"] == "abcdef1234"
    assert e[0]["records"] == {"claims": ["x-y"], "papers": ["p1"]}
    assert e[0]["pages"] == ["/applications/batteries/"]
