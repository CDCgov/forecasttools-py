import cfa.stf.forecasttools as ft


def test_get_us_loc_pop_tbl_reads_packaged_data():
    tbl = ft.get_us_loc_pop_tbl()

    assert tbl.columns == ["code", "abbr", "hrd", "name", "population"]
    assert tbl.get_column("abbr").to_list() == ft.LOCATION_LIST
    assert tbl.filter(tbl["abbr"] == "US").item(0, "name") == "United States"
