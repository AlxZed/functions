# Generated by nuclio.export.NuclioExporter

import mlrun

from cloudpickle import load
from typing import List
from sklearn.datasets import load_iris
import numpy as np

import warnings 
warnings.filterwarnings('ignore')

class ClassifierModel(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model('.pkl')
        self.model = load(open(model_file, 'rb'))

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""
        feats = np.asarray(body['inputs'])
        result: np.ndarray = self.model.predict(feats)
        return result.tolist()


from mlrun.runtimes import nuclio_init_hook
def init_context(context):
    nuclio_init_hook(context, globals(), 'serving_v2')

def handler(context, event):
    return context.mlrun_handler(context, event)
