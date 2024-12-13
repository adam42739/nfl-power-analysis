import nfl_data_py
import os
import pandas as pd
import numpy as np
from scipy import optimize
from typing import Callable


# ================== #
# 1. CACHE FUNCTIONS #
# ================== #


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


# ===================================== #
# 2. PERFORMANCE INDICATOR CALCULATIONS #
# ===================================== #

# =============================== #
# 2.1. Expected Score Differntial #
# =============================== #


def _init_exp_diff_frame(X: tuple[float], scores: pd.DataFrame):
    """
    Initialize the expected differential dataframe
    """
    exp_diff = pd.DataFrame(
        {"team": sorted(list(scores["home_team"].unique())), "exp_diff": [x for x in X]}
    )
    return exp_diff


def _exp_diff_error(
    exp_diff: pd.DataFrame, scores: pd.DataFrame, home_advan: float
) -> float:
    """
    Compute the mean error for the given, expected differential values, scores, and home field advantage
    """
    # create a table with the scores of each game and each teams expected differential
    table = pd.merge(left=scores, right=exp_diff, left_on="home_team", right_on="team")
    table = pd.merge(left=table, right=exp_diff, left_on="away_team", right_on="team")
    table = table.rename(
        {"exp_diff_x": "home_exp_diff", "exp_diff_y": "away_exp_diff"}, axis="columns"
    )
    table = table[["home_score", "away_score", "home_exp_diff", "away_exp_diff"]]

    # compute the expected score differential for each game
    game_exp_diff = table["home_exp_diff"] - table["away_exp_diff"] + home_advan

    # compute the real score differential for each game
    game_real_diff = table["home_score"] - table["away_score"]

    # compute and return the error in expected vs. real scores
    return np.sqrt(np.mean(np.square(game_exp_diff - game_real_diff)))


def _exp_diff_objective(
    X: tuple[float], scores: pd.DataFrame, home_advan: float
) -> float:
    """
    Define the objective function for minimization
    """
    exp_diff = _init_exp_diff_frame(X, scores)
    return _exp_diff_error(exp_diff, scores, home_advan)


def compute_exp_diff(pbp: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the expected differential table for the given season schedule
    """
    # compute the average home field advantage
    home_advan = np.mean(scores["home_score"]) - np.mean(scores["away_score"])

    # minimize the error
    solution = optimize.minimize(
        _exp_diff_objective,
        tuple(0 for _ in range(32)),
        args=(scores, home_advan),
        tol=1e-6,
    )

    # return the expected differential table
    exp_diff_table = _init_exp_diff_frame(solution.x, scores)
    return exp_diff_table


# ================ #
# 3. MODEL FITTING #
# ================ #


class PowerModel:
    def __init__(self):
        pass

    def load(self, fname: str):
        pass

    def dump(self, fname: str):
        pass

    def build(
        self,
        seasons: list[int],
        compute_functions: set[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = {
            compute_exp_diff
        },
    ):
        dfs = []
        for season in seasons:
            schedule = load_schedule(season)
            pbp = load_pbp(season)
            abbrs = sorted(
                list(pd.concat([schedule["home_team"], schedule["away_team"]]).unique())
            )
            df = pd.DataFrame(
                {
                    "team": [f"{season}{abbr}" for abbr in abbrs],
                    "abbr": [abbr for abbr in abbrs],
                }
            )
            for function in compute_functions:
                stat = function(pbp, schedule)
                df = pd.merge(left=df, right=stat, left_on="abbr", right_on="team")
                df = df.drop(columns="team_y")
                df = df.rename({"team_x": "team"}, axis="columns")
            dfs.append(df)
        df = pd.concat(dfs)


def create_power_model(seasons: list[int]):
    pass


model = PowerModel()
model.build([2022, 2023])
