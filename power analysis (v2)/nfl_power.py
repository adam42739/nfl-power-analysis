import nfl_data_py
import os
import pandas as pd
import numpy as np
from typing import Callable
from sklearn.linear_model import LinearRegression


# =============== #
# CACHE FUNCTIONS #
# =============== #


CACHE_PATH = os.path.join(os.path.dirname(__file__), "../cache")


def _pbp_path_name(season: int) -> str:
    return os.path.join(CACHE_PATH, f"season={season}/part.0.parquet")


def _schedule_path_name(season: int) -> str:
    return os.path.join(CACHE_PATH, f"schedule={season}.parquet")


def cache_data(seasons: list[int]):
    """
    Re-cache all data for the given seasons
    """
    # cache the play-by-play data
    nfl_data_py.cache_pbp(seasons, alt_path=CACHE_PATH)

    # cache schedules data
    for season in seasons:
        df = nfl_data_py.import_schedules([season])
        df.to_parquet(_schedule_path_name(season))


def load_pbp(season: int) -> pd.DataFrame:
    """
    Load play-by-play data from cache if possible, source if does not exist in cache
    """
    if os.path.exists(_pbp_path_name(season)):
        return nfl_data_py.import_pbp_data([season], cache=True, alt_path=CACHE_PATH)
    else:
        return nfl_data_py.import_pbp_data([season])


def load_schedule(season: int) -> pd.DataFrame:
    """
    Load schedules data from cache if possible, source if does not exist
    """
    if os.path.exists(_schedule_path_name(season)):
        return pd.read_parquet(_schedule_path_name(season))
    else:
        return nfl_data_py.import_schedules([season])


# ================= #
# UTILITY FUNCTIONS #
# ================= #


def differential_linreg(df: pd.DataFrame) -> pd.DataFrame:
    # compute the home field advantage
    home_advan = df["home_X"].mean() - df["away_X"].mean()

    # create the y (outcome) variable
    y = df["home_X"] - df["away_X"] + home_advan

    # team abbreviations alphabetically
    abbrs = sorted(pd.concat([df["home_team"], df["away_team"]]).unique())

    # create the X (feature) matrix
    df = df.reset_index(drop=True)
    X = np.zeros((len(y), 32))
    for i in range(len(y)):
        # set the home team's value to 1
        home_index = abbrs.index(df.loc[i, "home_team"])
        X[i, home_index] = 1

        # away team to -1
        away_index = abbrs.index(df.loc[i, "away_team"])
        X[i, away_index] = -1

    # run linear regression
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # return the KPI table
    kpi = pd.DataFrame({"abbr": abbrs, "kpi": model.coef_})
    return kpi


# ================================== #
# PERFORMANCE INDICATOR CALCULATIONS #
# ================================== #

# ========================== #
# Expected Score Differntial #
# ========================== #


def compute_exp_diff(pbp: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the expected differential table for the given season schedule
    """
    # get only the important columns and rename accordingly
    df = scores[["home_team", "away_team", "home_score", "away_score"]]
    df = df.rename({"home_score": "home_X", "away_score": "away_X"}, axis="columns")

    # return the differential linear regression KPI table
    return differential_linreg(df)


# ============= #
# MODEL FITTING #
# ============= #


class PowerModel:
    def __init__(self):
        pass

    def load(self, fname: str):
        pass

    def dump(self, fname: str):
        pass

    @staticmethod
    def _next_week_diff(
        row: pd.Series, schedule: pd.DataFrame, home_advan: float, df: pd.DataFrame
    ) -> float | None:
        """
        Helper function that computes the teams next week score differential and returns its opponents KPIs
        """
        # get all possible future games for the team
        this_team = (schedule["home_team"] == row["abbr"]) | (
            schedule["away_team"] == row["abbr"]
        )
        after_week = schedule["week"] > row["week"]
        possible_games: pd.DataFrame = schedule[this_team & after_week]

        if not possible_games.empty:
            # get the next weeks game as a pd.Series
            first_week = possible_games["week"].min()
            next_week = possible_games[possible_games["week"] == first_week].iloc[0]

            # compute the differential (correcting for home field advantage) and get the oppositions KPIs
            if next_week["home_team"] == row["abbr"]:
                # compute the score differential
                score_diff = (
                    next_week["home_score"] - next_week["away_score"] - (home_advan / 2)
                )

                # get the oppositions KPIs
                df_oppo: pd.DataFrame = df[df["abbr"] == next_week["away_team"]]
                kpis = df_oppo.filter(like="kpi", axis=1).values[0]

                return pd.Series(
                    list(kpis) + [score_diff],
                    index=list(row.filter(like="kpi").index) + ["y"],
                )
            else:
                score_diff = (
                    next_week["away_score"] - next_week["home_score"] + (home_advan / 2)
                )

                # get the oppositions KPIs
                df_oppo: pd.DataFrame = df[df["abbr"] == next_week["home_team"]]
                kpis = df_oppo.filter(like="kpi", axis=1).values[0]

                return pd.Series(
                    list(kpis) + [score_diff],
                    index=list(row.filter(like="kpi").index) + ["y"],
                )

        else:
            return pd.Series(
                [None for _ in range(len(row.filter(like="kpi").index) + 1)],
                index=list(row.filter(like="kpi").index) + ["y"],
            )

    @staticmethod
    def _create_dataset_singleweek(
        season: int,
        week: int,
        schedule: pd.DataFrame,
        pbp: pd.DataFrame,
        compute_functions: set[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]],
    ):
        """
        Compute the KPIs for a single week
        """
        # trim the input dataframes to only have data up until this week
        up_to_week = schedule["week"] <= week
        schedule_trim = schedule[up_to_week]
        up_to_week = pbp["week"] <= week
        pbp_trim = pbp[up_to_week]

        # get a list of all team abbreviations sorted
        abbrs = sorted(
            pd.concat([schedule["home_team"], schedule["away_team"]]).unique()
        )

        # initialize the dataframe formatted like:
        #   |                        index |           abbr |   season |   week |
        #   |------------------------------|----------------|----------|--------|
        #   | {season}{abbreviation}{week} | {abbreviation} | {season} | {week} |
        #   |               .              |        .       |    .     |   .    |
        #   |               .              |        .       |    .     |   .    |
        #   |               .              |        .       |    .     |   .    |
        df = pd.DataFrame(
            {
                "abbr": [abbr for abbr in abbrs],
                "season": [season for _ in abbrs],
                "week": [week for _ in abbrs],
            },
            index=[f"{season}{abbr}{week}" for abbr in abbrs],
        )

        # iterate over each compute function to populate the
        # dataframe with predictors horizontally
        kpi_index = 0
        for function in compute_functions:
            stat = function(pbp_trim, schedule_trim)
            df = df.reset_index().merge(stat, how="left", on="abbr").set_index("index")
            df = df.rename({"kpi": f"kpi_{kpi_index}"}, axis="columns")
            kpi_index += 1

        # compute the home field advantage
        home_advan = (
            schedule_trim["home_score"].mean() - schedule_trim["away_score"].mean()
        )

        # compute each team's score differential for the following week (NA is does not exist)
        df[[f"opp_kpi_{i}" for i in range(len(compute_functions))] + ["y"]] = df.apply(
            PowerModel._next_week_diff,
            axis=1,
            schedule=schedule,
            home_advan=home_advan,
            df=df,
        )

        return df

    def build(
        self,
        seasons: list[int],
        compute_functions: list[
            Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
        ] = [
            compute_exp_diff,
        ],
    ):
        """
        Build the power model
        """
        self.compute_functions = compute_functions

        # compute all the KPIs for each season and week
        dfs = []
        for season in seasons:
            # load in the schedule and pbp data for the given season
            schedule = load_schedule(season)
            pbp = load_pbp(season)

            # compute which week was the last week of the season (or last played week for the current season)
            max_week = int(schedule["week"].max())

            # iterate over each week
            for week in range(1, max_week + 1):
                df = PowerModel._create_dataset_singleweek(
                    season, week, schedule, pbp, compute_functions
                )
                dfs.append(df)
        df = pd.concat(dfs)

        # drop NAs (datapoints where the team played no more future games)
        df = df.dropna()

        # get X and y as ndarray
        X_1 = df.filter(regex="^kpi", axis=1).values
        X_2 = df.filter(like="opp", axis=1).values
        X = X_1 - X_2
        y = df["y"].values

        # run the model
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(X, y)

    def compute_power(
        self, pbp: pd.DataFrame, schedule: pd.DataFrame, season: int, week: int
    ) -> pd.DataFrame:
        """
        Compute the power scores by predicting from the model
        """
        # get the KPIs for the given data
        df = PowerModel._create_dataset_singleweek(
            season, week, schedule, pbp, self.compute_functions
        )
        X = df.filter(regex="^kpi", axis=1).values
        X = X - np.mean(X, axis=0)

        # use the model to predict the power score
        y = self.model.predict(X)

        # create and return the power table
        power_table = pd.DataFrame({"team": df["abbr"], "power": y})
        return power_table


# TODO
#   - read/write IO for the model
#   - look into the probabalistic adjustment


model = PowerModel()
model.build([2023])
power_table = model.compute_power(load_pbp(2024), load_schedule(2024), 2024, 12)
print(power_table.sort_values(by="power", ascending=False))
