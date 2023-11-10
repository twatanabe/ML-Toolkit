import lightgbm as lgb
import optuna.integration.lightgbm as lgbo
from sklearn.model_selection import train_test_split


def make_lgb_data(test_size, random_state, metric, X, value):
    X_train, X_test, t_train, t_test = train_test_split(
        X, value, test_size=test_size, random_state=random_state
    )

    lgb_train = lgb.Dataset(X_train, t_train)

    lgb_eval = lgb.Dataset(X_test, t_test, reference=lgb_train)

    dic_return = {
        "X_train": X_train,
        "X_test": X_test,
        "t_train": t_train,
        "t_test": t_test,
        "lgb_train": lgb_train,
        "lgb_eval": lgb_eval,
    }

    return dic_return


def tune_params(test_size, random_state, objective, metric, X, value):
    opt_params = {
        "force_row_wise": True,
        "force_col_wise": False,
        "objective": objective,
        "metric": metric,
    }

    dic = make_lgb_data(test_size, random_state, metric, X, value)
    lgb_train = dic["lgb_train"]
    lgb_eval = dic["lgb_eval"]

    opt = lgbo.train(
        opt_params,
        lgb_train,
        valid_sets=lgb_eval,
        verbose_eval=False,
        num_boost_round=10,
        early_stopping_rounds=10,
    )

    return opt
