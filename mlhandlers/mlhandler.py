import mlrun
import warnings
import json

from typing import List
from sklearn.model_selection import train_test_split
from importlib import import_module
from inspect import _empty, signature
from mlrun.utils.helpers import create_class
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
from mlrun.frameworks.auto_mlrun import AutoMLRun

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


def create_class(pkg_class: str):
    """Create a class from a package.module.class string

    :param pkg_class:  full class location,
                       e.g. "sklearn.model_selection.GroupKFold"
    """
    splits = pkg_class.split(".")
    clfclass = splits[-1]
    pkg_module = splits[:-1]
    class_ = getattr(import_module(".".join(pkg_module)), clfclass)
    return class_


def get_class_fit(module_pkg_class: str):
    """generate a model config
    :param module_pkg_class:  str description of model, e.g.
        `sklearn.ensemble.RandomForestClassifier`
    """
    splits = module_pkg_class.split(".")
    model_ = getattr(import_module(".".join(splits[:-1])), splits[-1])
    f = list(signature(model_().fit).parameters.items())
    d = {}
    for i in range(len(f)):
        d.update({f[i][0]: None if f[i][1].default is _empty else f[i][1].default})

    return {
        "CLASS": model_().get_params(),
        "FIT": d,
        "META": {
            "pkg_version": import_module(splits[0]).__version__,
            "class": module_pkg_class,
        },
    }


def update_model_config(model_pkg, skparams):
    """generate an sklearn model configuration

    input can be either a "package.module.class" or
    a json file
    """
    if model_pkg.endswith("json"):
        model_config = json.load(open(model_pkg, "r"))
    else:
        model_config = get_class_fit(model_pkg)

    # we used to use skparams as is (without .items()) so supporting both cases for backwards compatibility
    skparams = skparams.items() if isinstance(skparams, dict) else skparams
    for k, v in skparams:
        if k.startswith("CLASS_"):
            model_config["CLASS"][k[6:]] = v
        if k.startswith("FIT_"):
            model_config["FIT"][k[4:]] = v

    return model_config


def _gen_lgbm_model(model_type: str, lgbm_params: dict):
    mtypes = {
        "lgbm.classifier": "lightgbm.LGBMClassifier",
        "lgbm.ranker": "lightgbm.LGBMRanker",
        "lgbm.regressor": "lightgbm.LGBMRegressor",
    }
    if model_type.endswith("json"):
        model_config = model_type
    elif model_type in mtypes.keys():
        model_config = mtypes[model_type]
    else:
        raise Exception("unrecognized model type, see help documentation")

    return update_model_config(model_config, lgbm_params)


def _gen_xgb_model(model_type: str, xgb_params: dict):
    """generate an xgboost model

    Multiple model types that can be estimated using
    the XGBoost Scikit-Learn API.

    Input can either be a predefined json model configuration or one
    of the five xgboost model types: "classifier", "regressor", "ranker",
    "rf_classifier", or "rf_regressor".

    In either case one can pass in a params dict to modify defaults values.

    Based on `mlutils.models.gen_sklearn_model`, see the function
    `sklearn_classifier` in this repository.

    :param model_type: one of "classifier", "regressor",
                       "ranker", "rf_classifier", or
                      "rf_regressor"
    :param xgb_params: class init parameters
    """
    mtypes = {
        "xgb.classifier": "xgboost.XGBClassifier",
        "xgb.regressor": "xgboost.XGBRegressor",
        "xgb.ranker": "xgboost.XGBRanker",
        "xgb.rf_classifier": "xgboost.XGBRFClassifier",
        "xgb.rf_regressor": "xgboost.XGBRFRegressor",
    }
    if model_type.endswith("json"):
        model_config = model_type
    elif model_type in mtypes.keys():
        model_config = mtypes[model_type]
    else:
        raise Exception("unrecognized model type, see help documentation")

    return update_model_config(model_config, xgb_params)


def _gen_sklearn_model(model_pkg, skparams):
    if model_pkg.endswith("json"):
        model_config = json.load(open(model_pkg, "r"))
    else:
        model_config = get_class_fit(model_pkg)

    # we used to use skparams as is (without .items()) so supporting both cases for backwards compatibility
    skparams = skparams.items() if isinstance(skparams, dict) else skparams

    for k, v in skparams:
        if k.startswith("CLASS_"):
            model_config["CLASS"][k[6:]] = v
        if k.startswith("FIT_"):
            model_config["FIT"][k[4:]] = v

    return model_config


def _gen_model_configuration(model_pkg, model_params):
    """
    Generate a model configuration
    input can be either a "package.module.class" or a json file
    """
    if "sklearn" in model_pkg:
        model_config = _gen_sklearn_model(model_pkg, model_params)

    elif "xgb" in model_pkg:
        model_config = _gen_xgb_model(model_pkg, model_params)

    elif "lgbm" in model_pkg:
        model_config = _gen_lgbm_model(model_pkg, model_params)

    else:
        raise AttributeError(
            "The model passed does not belong to sklearn, xgb, or lgbm"
        )
    return model_config


def train(
    context: MLClientCtx,
    dataset: DataItem,
    model_class: str,
    label_column: str = "label",
    model_name: str = "trained_model",
    test_size: float = 0.2,
    artifacts: List[str] = [],
    save_format: str = "pkl",
):
    """ """

    # Set model config file
    model_config = _gen_model_configuration(model_class, context.parameters.items())

    # Pull DataFrame from DataItem
    dataset = (
        dataset.as_df() if type(dataset) == mlrun.datastore.base.DataItem else dataset
    )

    # Split according to test_size
    X = dataset[dataset.columns[dataset.columns != label_column]]
    y = dataset[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Update config with the new split
    model_config["FIT"].update({"X": X_train, "y": y_train})

    # ?
    model_class = create_class(model_config["META"]["class"])
    model = model_class(**model_config["CLASS"])

    # Wrap our model with Mlrun features, specify the test dataset for analysis and accuracy measurements
    AutoMLRun.apply_mlrun(
        model=model, context=context, x_validation=X_test, y_validation=y_test
    )

    # Train our model
    model.fit(model_config["FIT"]["X"], model_config["FIT"]["y"])


def evaluate(
    context: MLClientCtx,
    dataset: DataItem,
    model_path: str,
    artifacts: List[str],
    label_column: str = "label",
):
    # Pull DataFrame from DataItem
    dataset = (
        dataset.as_df() if type(dataset) == mlrun.datastore.base.DataItem else dataset
    )

    model_handler = AutoMLRun.load_model(model_path)

    X_test = dataset[dataset.columns[dataset.columns != label_column]]
    y_test = dataset[label_column]

    AutoMLRun.apply_mlrun(model_handler.model, y_test=y_test, model_path=model_path)

    model_handler.model.predict(X_test)


def predict(
    context: MLClientCtx,
    dataset: DataItem,
    model_path: str,
    artifacts: List[str],
    label_column: str = None,
):
    # Pull DataFrame from DataItem
    dataset = (
        dataset.as_df() if type(dataset) == mlrun.datastore.base.DataItem else dataset
    )

    if label_column:
        dataset = dataset[dataset.columns[dataset.columns != label_column]]

    model_handler = AutoMLRun.load_model(model_path)
    model_handler.model.predict(dataset)
