import logging
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import check_random_state
from fairlearn.datasets import fetch_adult, fetch_boston
from folktables import ACSDataSource, ACSIncome
from frwo.utils import *


def generate_dataset(dataset_name, n_samples: int | None = None, random_state=None):
    """
    Helper method that returns a dataset based on the specified label.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    n_samples : int | None
        Number of samples to return from the dataset. If None, use the whole dataset.
    random_state : int, numpy.random.RandomState or None, default None
        What the dataset's random draws -- sampling `n_samples` rows, and
        undersampling a class -- are made from. None draws from the global
        numpy generator, so the dataset depends on what was drawn before it.

    Returns
    -------
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
    """
    if dataset_name == "Adult":
        X, y, feature_types = generate_adult_dataset(
            n_samples, random_state=random_state
        )
    elif dataset_name == "COMPAS":
        X, y, feature_types = generate_compas_dataset(
            n_samples, random_state=random_state
        )
    elif dataset_name == "Boston":
        X, y, feature_types = generate_boston_housing_dataset(
            n_samples, random_state=random_state
        )
    elif "ACSIncome__" in dataset_name:
        state = dataset_name[-2:]
        X, y, feature_types = generate_acs_income(
            n_samples, state=state, random_state=random_state
        )
    else:
        raise ValueError(f"Unrecognized dataset name: {dataset_name}")

    return X, y, feature_types


def generate_adult_dataset(
    n: int | None = None,
    z_col="is_race_white",
    y_col="is_income_over_50k",
    random_state=None,
):
    """
    Wrapper function for generating a sample of the adult dataset. This
    includes sampling down to just `n` samples, specifying the protected
    attribute 'z',

    Parameters
    ---------
    n : int, default None
        Number of records to sample from dataset. If None, use the whole dataset.
    z_col : str, default 'is_race_white'
        The column to use as the protected attribute. Must be binary.
    y_col : str, default 'is_income_over_50k'
        The column to use as the target variable. Must be binary.
    random_state : int, numpy.random.RandomState or None, default None
        What the random draws are made from. See `generate_dataset()`.

    Returns
    -------
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
    """
    data = fetch_adult(as_frame=True)
    df = data.data.copy()
    df["income"] = data.target.copy()
    use_legacy_str_dtype(df)

    # Take sample if possible
    rng = check_random_state(random_state)
    if n is not None and n < len(df):
        df = df.sample(n, random_state=rng)

    # Common transformations
    df["is_income_over_50k"] = df["income"] == ">50K"
    df["is_race_white"] = df["race"] == "White"

    # Specify the target variable `y`
    df["y"] = df[y_col].astype(int)

    # Specify the protected attribute `z`
    df["z"] = df[z_col].astype(int)

    # Display useful summary debug on z and y
    logging.debug("Dataset count of each z, y group")
    logging.debug(df_to_log(df.groupby(["z"])[["y"]].agg(["count", "mean"])))

    quantile_features = []
    for cont_feat in [
        "age",
        # 'educational-num',
        "capital-gain",
        "capital-loss",
        # 'hours-per-week',
    ]:
        for q in [
            0.1,
            0.75,
            0.9,
        ]:
            f = f"{cont_feat}__{q}"
            df[f] = df[cont_feat] <= df[cont_feat].quantile(q)
            quantile_features.append(f)

    # Split into inputs and target variables
    y = df["y"]
    X = df.copy().drop(columns=["y", y_col, z_col, "income"])

    # NOTE: 05/20/2023
    # Resampling messes up the feature expectations. Don't do this if you're
    # doing IRL.
    #
    # Balance the positive and negative classes
    rus = RandomUnderSampler(sampling_strategy=0.42, random_state=rng)
    X, y = rus.fit_resample(X, y)

    feature_types = {
        "boolean": [
            "z",
        ]
        + quantile_features,
        "categoric": [
            "workclass",
            "education",
            "marital-status",
            # 'occupation',
            "relationship",
            "native-country",
            # 'race',
            "sex",
        ],
        "continuous": [
            # 'age',
            # 'educational-num',
            # 'capital-gain',
            # 'capital-loss',
            # 'hours-per-week',
        ],
        "meta": ["fnlwgt"],
        "hidden": [],
    }

    return X, y, feature_types


def generate_compas_dataset(
    n: int | None = None,
    z_col="is_race_white",
    y_col="is_recid",
    filepath="./data/compas/cox-violent-parsed.csv",
    random_state=None,
):
    """
    Wrapper function for generating a sample of the Compas dataset. This
    includes sampling down to just `n` samples, specifying the protected
    attribute 'z',

    Parameters
    ---------
    filepath : str
        Filepath for dataset.
    n : int, default None
        Number of records to sample from dataset. If None, use the whole dataset.
    z_col : str, default 'is_race_white'
        The column to use as the protected attribute. Must be binary.
    y_col : str, default 'is_recid'
        The column to use as the target variable. Must be binary.
    random_state : int, numpy.random.RandomState or None, default None
        What the random draws are made from. See `generate_dataset()`.

    Returns
    -------
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.


    Dataset column descriptions:
        https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1452
    """

    # Import dataset
    df = pd.read_csv(filepath)
    use_legacy_str_dtype(df)

    # Take sample if possible
    rng = check_random_state(random_state)
    if n is not None and n < len(df):
        df = df.sample(n, random_state=rng)

    # Filter out records where we don't know their compas risk score
    df = df.query("is_recid >= 0").copy()

    # Common transformations
    df["is_race_white"] = (df["race"] == "Caucasian").astype(int)
    df = df.rename(
        columns={
            "sex": "gender",
        }
    )

    # Specify the target variable `y`
    df["y"] = df[y_col].astype(int)

    # Specify the protected attribute `z`
    df["z"] = df[z_col].astype(int)

    # Display useful summary debug on z and y
    logging.debug("Dataset count of each z, y group")
    logging.debug(df_to_log(df.groupby(["z"])[["y"]].agg(["count", "mean"])))

    # # Balance the two protected groups
    # # Split into inputs and target variables
    # z = df['z']
    # X = df.copy().drop(columns=[z_col])
    # rus = RandomUnderSampler(sampling_strategy=1)
    # X, z = rus.fit_resample(X, z)
    # df = X.copy()
    # df['z'] = z

    quantile_features = []
    for cont_feat in [
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
    ]:
        for q in [
            0.1,
            0.75,
            0.9,
        ]:
            f = f"{cont_feat}__{q}"
            df[f] = df[cont_feat] <= df[cont_feat].quantile(q)
            quantile_features.append(f)

    # Split into inputs and target variables
    y = df["y"]
    X = df.copy().drop(columns="y")

    feature_types = {
        "boolean": [
            "z",
        ]
        + quantile_features,
        "categoric": [
            "age_cat",
            "c_charge_degree",
            "gender",
        ],
        "continuous": [
            # 'age',
            # 'juv_fel_count',
            # 'juv_misd_count',
            # 'juv_other_count',
            # 'priors_count',
        ],
        "meta": [],
        "hidden": [
            "decile_score",
            "v_decile_score",
            "score_text",
            "v_score_text",
        ],
    }

    return X, y, feature_types


def generate_boston_housing_dataset(n: int | None = None, random_state=None):
    """
    Wrapper function for generating a sample of the boston housing dataset.

    Parameters
    ---------
    n : int, default None
        Number of records to sample from dataset. If None, use the whole dataset.
    random_state : int, numpy.random.RandomState or None, default None
        What the random draws are made from. See `generate_dataset()`.

    Returns
    -------
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
    """
    data = fetch_boston(as_frame=True)
    df = data.data.copy()
    df["LSTAT_binary"] = df["LSTAT"] >= df["LSTAT"].median()
    df["MEDV"] = data.target.copy()
    use_legacy_str_dtype(df)

    # Take sample if possible
    rng = check_random_state(random_state)
    if n is not None and n < len(df):
        df = df.sample(n, random_state=rng)
    if n is not None and n > len(df):
        df = df.sample(n, replace=True, random_state=rng)

    # Specify the protected attribute `z`
    # Median value for Z
    df["z"] = (df["B"] >= 381.44).astype(int)

    quantile_features = []
    for cont_feat in [
        # 'B',
        "CRIM",
        "ZN",
        "RM",
        "LSTAT",
    ]:
        for q in [
            # .05,
            # .1,
            0.25,
            0.5,
            0.75,
        ]:
            f = f"{cont_feat}__{q}"
            df[f] = df[cont_feat] <= df[cont_feat].quantile(q)
            quantile_features.append(f)

    y = (df["MEDV"] >= df["MEDV"].quantile(0.75)).astype(int).copy()
    X = df.drop(columns="MEDV")

    # Balance the positive and negative classes
    rus = RandomUnderSampler(sampling_strategy=1, random_state=rng)
    X, y = rus.fit_resample(X, y)
    feature_types = {
        "boolean": [
            "z",
        ]
        + quantile_features,
        "categoric": [],
        "continuous": [
            # 'CRIM',  # per capita crime rate by town
            # 'ZN',  # prop of residential land zoned for lots over 25,000 sqft
            # 'INDUS',  # prop of non-retail business acres per town
            # 'CHAS',  # Charles River dummy var (= 1 if bounds river; else 0)
            # 'NOX',  # nitric oxides concentration (parts per 10 million)
            # 'RM',  # average number of rooms per dwelling
            # 'AGE',  # proportion of owner-occupied units built prior to 1940
            # 'DIS',  # weighted distances to five Boston employment centers
            # 'RAD',  # index of accessibility to radial highways
            # 'TAX',  # full-value property-tax rate per $10,000
            # 'PTRATIO',  # pupil-teacher ratio by town
            # 'B', # 1000(Bk - 0.63)^2 where Bk is the proportion of Black ppl
            # 'LSTAT',  # % lower status of the population
        ],
        "meta": [],
        "target": [
            "MEDV",  # Median value of owner-occupied homes in $1000's],
        ],
        "hidden": [],
    }

    return X, y, feature_types


def generate_acs_income(n: int | None = None, state=None, random_state=None):
    """
    Wrapper function for generating a sample of the folktable ACSIncome
    dataset.

    See Appendix B of https://arxiv.org/pdf/2108.04884.pdf for full feature
    details.

    Parameters
    ---------
    n : int, default None
        Number of records to sample from dataset. If None, use the whole dataset.
    state : str
        The US state to use.
    random_state : int, numpy.random.RandomState or None, default None
        What the random draws are made from. See `generate_dataset()`.

    Returns
    -------
    X : pandas.DataFrame
        The X (including z) columns.
    y : pandas.Series
        Just the y column.
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
    """
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    data = data_source.get_data(states=[state], download=True)
    X, y, _ = ACSIncome.df_to_pandas(data)
    df = X.copy()
    df["y"] = y
    df["y"] = df["y"].astype(int)
    del X, y
    use_legacy_str_dtype(df)

    # Take sample if possible
    rng = check_random_state(random_state)
    if n is not None and n < len(df):
        df = df.sample(n, random_state=rng)
    if n is not None and n > len(df):
        df = df.sample(n, replace=True, random_state=rng)

    # Specify the protected attribute `z`
    z_col = "RAC1P"
    df["z"] = (df[z_col] == 1).astype(int)

    df = df.fillna(-1)

    # Features
    # --------
    # AGEP (Age) : 0-99 integers., nullable
    # COW (Class of worker): 1-9 integers, nullable
    # SCHL (Educational attainment) : 1-24 integers, nullable
    # MAR (Marital status) : 1-5 integers, not nullable
    # OCCP (Occupation) : categoric, 529 distinct values
    # POBP (Place of birth) : categoric, 219 distinct values
    # WKHP (Usual hours worked per week) : 1-99, nullable
    # SEX (Sex) : integers, 1 Male, 2 Female
    # RAC1P (Recoded detailed race code)
    #   - 1: White alone
    #   - 2: Black or African American alone
    #   - 3: American Indian alone
    #   - 4: Alaska Native alone
    #   - 5: American Indian and Alaska Native tribes specified
    #   - 6: Asian alone
    #   - 7: Native Hawaiian and Other Pacific Islander alone
    #   - 8: Some Other Race alone
    #   - 9: Two or More Races
    quantile_features = []
    for cont_feat in [
        "AGEP",
        # 'SCHL',
        # 'WKHP',
    ]:
        for q in [
            0.05,
            0.15,
            0.5,
            0.85,
            0.95,
        ]:
            f = f"{cont_feat}__{q}"
            df[f] = df[cont_feat] <= df[cont_feat].quantile(q)
            quantile_features.append(f)

    feature_types = {
        "boolean": [
            "z",
            "SEX",
        ]
        + quantile_features,
        "categoric": [
            "COW",
            "MAR",
            # 'OCCP',
            "POBP",
            "RAC1P",
        ],
        "continuous": [],
        "meta": [],
        "target": [],
        "hidden": [],
    }

    # Split into inputs and target variables
    y = df["y"]
    X = df.copy().drop(columns=["y"])
    del df

    # NOTE: 05/20/2023
    # Resampling messes up the feature expectations. Don't do this if you're
    # doing IRL.
    #
    # Balance the positive and negative classes
    # rus = RandomUnderSampler(sampling_strategy=1)
    # X, y = rus.fit_resample(X, y)

    return X, y, feature_types
