from functools import cache
from importlib.resources import as_file, files

__all__ = ["get_us_loc_pop_tbl", "LOCATION_LIST"]


@cache
def get_us_loc_pop_tbl():
    import polars as pl

    resource = files(__package__).joinpath("data", "us_location_pop.csv")
    with as_file(resource) as path:
        return pl.read_csv(path)


LOCATION_LIST = get_us_loc_pop_tbl().get_column("abbr").to_list()
