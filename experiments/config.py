"""Shared experiment configuration defaults."""

BASE_FEATURES = (
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "temperature",
)

GLOBAL_FEATURES = (
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "location_id",
    "temperature",
)

TARGET_COLUMN = "consumption"

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_OUTPUT_DIR = "results/experiments"

DEFAULT_LOCATION_NAMES = {
    0: "Helsingfors",
    1: "Oslo",
    2: "Stavanger",
    3: "Trondheim",
    4: "Tromso",
    5: "Bergen",
}

DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42,
}

DEFAULT_NUM_BOOST_ROUND = 1000
DEFAULT_EARLY_STOPPING_ROUNDS = 50
