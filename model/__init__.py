from .model_103 import Model as Model_103
from .model import Model as Model

dictionary = {
    "m_A": Model,
    "m_B": Model,
    "m_X_103": Model_103,
}

def get_model(model_name: str):
    return dictionary[model_name]