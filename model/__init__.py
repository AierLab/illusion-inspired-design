from .model_103 import Model as Model_103
from .model_202 import Model as Model_202
from .model_102 import Model as Model_102
from .model import Model as Model
from .model_102_v2 import Model as Model_102_v2
from .model_102_v3 import Model as Model_102_v3
from .model_comp import ModelComp
from .model_comp_reverse import ModelCompReverse
from .model_vae import ResNetVAE

dictionary = {
    "m_A": Model,
    "m_B": Model,
    "m_X_102": Model_102,
    "m_X_202": Model_202,
    "m_X_103": Model_103,
    "m_X_102_v2": Model_102_v2,
    "m_X_102_v3": Model_102_v3,
    "m_X_comp": ModelComp,
    "m_X_comp_reverse": ModelCompReverse,
    "m_VAE": ResNetVAE
}

def get_model(model_name: str):
    return dictionary[model_name]