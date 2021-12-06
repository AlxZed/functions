from mlrun.datastore import DataItem
from typing import List
from mlrun.execution import MLClientCtx
from mlrun.mlutils.models import gen_sklearn_model
from mlrun.utils.helpers import create_class
from mlrun.artifacts.model import ModelArtifact
from sklearn.model_selection import train_test_split
from mlrun.frameworks.sklearn import apply_mlrun

import mlrun
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def train_model(context: MLClientCtx,
                dataset: DataItem,
                model_class: str,
                model_name: str,
                label_column: str = 'label',
                test_size: float = 0.2,
                artifacts: List[str] = [],
                save_format: str = 'pkl',
                ):
    
    print(type(dataset))
    # set model config file
    model_config = gen_sklearn_model(model_class, context.parameters.items())
    
    # Pull DataFrame from DataItem
    if mlrun.datastore.base.DataItem:
        dataset = dataset.as_df()
    
    # Split according to test_size
    X = dataset.loc[:, dataset.columns != label_column]
    y = dataset[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
          
    # Update config with the new split
    model_config["FIT"].update({"X": X_train, "y": y_train.values})

    
    #?
    model_class = create_class(model_config["META"]["class"])
    
    #load model with params?
    model = model_class(**model_config["CLASS"])    
    
    # Wrap our model with Mlrun features, specify the test dataset for analysis and accuracy measurements
    apply_mlrun(model, model_name=model_name, X_test=X_test, y_test=y_test)
    
    # Train our model
    model.fit(model_config["FIT"]["X"],model_config["FIT"]["y"])

#     artifact_path = context.artifact_subpath(models_dest)
#     plots_path = context.artifact_subpath(models_dest, plots_dest)
    
#     return ModelArtifact, artifacts
