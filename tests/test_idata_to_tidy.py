import polars as pl
import pytest

import forecasttools


class SimpleInferenceData:
    """
    Test class for a simple arviz-like
    InferenceData object. Such a class
    needs only have groups and a
    representation as a polars dataframe.
    """

    def __init__(self, groups: list[str], df: pl.DataFrame) -> None:
        self._groups = groups
        self._df = df

    def groups(self) -> list[str]:
        """
        Retrieves member group names from
        SimpleInferenceData.
        """
        return self._groups

    def to_dataframe(self) -> pl.DataFrame:
        """
        Provides representation of
        SimpleInferenceData as a polars
        dataframe.
        """
        return self._df


@pytest.fixture
def simple_inference_data():
    """
    Provides a SimpleInferenceData object for
    testing with a small Polars DataFrame. This
    fixture returns a SimpleInferenceData object
    that simulates an arviz-like InferenceData with
    two chains, each having a different number of
    draws (chain 0 has 2 draws, chain 1 has 3 draws).
    It also includes two groups: "posterior" and
    "sample", each with one or more parameters.

    Returns
    -------
    SimpleInferenceData
        An object containing the specified groups and
        a Polars DataFrame that mimics the structure of
        an arviZ InferenceData when converted to a
        DataFrame.
    """
    data = {
        # notice the different number of draws per chain.
        "chain": [0, 0, 1, 1, 1],
        "draw": [0, 1, 0, 1, 2],
        # this is what the columns might look like when converting an
        # inferenceData object to a Polars DataFrame.
        "('posterior', alpha)": [1, 2, 3, 4, 5],
        "('posterior', beta)": [10, 20, 30, 40, 50],
        # arbitrary integer values for the "sample" group.
        "('sample', gamma)": [100, 200, 300, 400, 500],
    }
    df = pl.DataFrame(data)
    groups = ["posterior", "sample"]
    return SimpleInferenceData(groups=groups, df=df)


def test_invalid_group(simple_inference_data):
    """
    Tests that passing an invalid group name
    to the conversion function raises a ValueError.

    Parameters
    ----------
    simple_inference_data : SimpleInferenceData
        A fixture providing a test
        InferenceData-like object with two groups.
    """
    with pytest.raises(ValueError) as out_error:
        _ = forecasttools.convert_inference_data_to_tidydraws(
            simple_inference_data, groups=["nonexistent_group"]
        )
    assert "not found" in str(out_error.value)


def test_valid_groups(simple_inference_data):
    """
    Tests that requesting a valid group
    ('posterior') returns a dictionary containing
    a Polars DataFrame for that group.

    Parameters
    ----------
    simple_inference_data : SimpleInferenceData
        A fixture providing a test
        InferenceData-like object with two groups.
    """
    result = forecasttools.convert_inference_data_to_tidydraws(
        simple_inference_data, groups=["posterior"]
    )
    assert isinstance(result, dict)
    assert "posterior" in result
    assert isinstance(result["posterior"], pl.DataFrame)


def test_default_groups(simple_inference_data):
    """
    Tests that when no specific groups are passed
    (i.e., groups=None), all groups from the
    InferenceData-like object are processed.

    Parameters
    ----------
    simple_inference_data : SimpleInferenceData
        A fixture providing a test InferenceData-like
        object with two groups.
    """
    result = forecasttools.convert_inference_data_to_tidydraws(
        simple_inference_data, groups=None
    )
    assert set(result.keys()) == set(simple_inference_data.groups())


def test_tidydraws_format(simple_inference_data):
    """
    Tests that the output for the 'posterior' group
    is in the correct tidy draws format expected by
    tidybayes. Specifically, check: required columns
    (.chain, .iteration, .draw) are present,
    parameter names (alpha, beta) are cleaned up,
    the number of rows matches the total draws across
    chains, the .chain, .iteration, and .draw columns
    are incremented and sorted as expected.

    Parameters
    ----------
    simple_inference_data : SimpleInferenceData
        A fixture providing a test InferenceData-like
        object with two groups.
    """
    result = forecasttools.convert_inference_data_to_tidydraws(
        simple_inference_data, groups=["posterior"]
    )
    df = result["posterior"]

    # check that the tidybayes required columns are present.
    for col in [".chain", ".iteration", ".draw"]:
        assert col in df.columns

    # expected sorted order after successful processing
    expected = [
        {".chain": 1, ".iteration": 1, ".draw": 1, "alpha": 1, "beta": 10},
        {".chain": 1, ".iteration": 2, ".draw": 2, "alpha": 2, "beta": 20},
        {".chain": 2, ".iteration": 1, ".draw": 3, "alpha": 3, "beta": 30},
        {".chain": 2, ".iteration": 2, ".draw": 4, "alpha": 4, "beta": 40},
        {".chain": 2, ".iteration": 3, ".draw": 5, "alpha": 5, "beta": 50},
    ]
    df_sorted = df.sort([".chain", ".iteration"])
    rows = df_sorted.select(
        [".chain", ".iteration", ".draw", "alpha", "beta"]
    ).to_dicts()

    for row, exp in zip(rows, expected):
        assert row[".chain"] == exp[".chain"]
        assert row[".iteration"] == exp[".iteration"]
        assert row[".draw"] == exp[".draw"]
        assert row["alpha"] == exp["alpha"]
        assert row["beta"] == exp["beta"]
    # total rows equal sum of draws across chains
    assert df.height == 5
