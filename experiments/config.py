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
    "num_leaves": 35,
    "learning_rate": 0.03,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 19,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "seed": 42,
}

LIGHTGBM_GLOBAL_PARAM_GRID = {
    "num_leaves": [35, 40, 45],
    "learning_rate": [0.03, 0.02],
    "min_child_samples": [18,19],
    "feature_fraction": [1.0],
}

DEFAULT_NUM_BOOST_ROUND = 1000
DEFAULT_EARLY_STOPPING_ROUNDS = 50

DEFAULT_RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
}

RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [10, None],
    "min_samples_split": [2, 10],
}


# best performance will likely be when the step size and test size are 1
# but this is also computationally expensive
# TODO: find a good balance between performance and runtime by using a larger step size and test size
DEFAULT_TRAIN_SIZE_ROLLING_WINDOW = 5 * 24 * 20  # 5 cities, 24 hours, 20 days
DEFAULT_VAL_SIZE_ROLLING_WINDOW = 5 * 24 * 3     # 5 cities, 24 hours, 3 days
DEFAULT_TEST_SIZE_ROLLING_WINDOW = 5 * 24 * 1    # 5 cities, 24 hours, 1 day
DEFAULT_STEP_SIZE_ROLLING_WINDOW = 5 * 24 * 1    # 5 cities, 24 hours, 1 day step

