from .model_103 import Model as Model_103
from .model_102 import Model as Model_102
from .model import Model as Model
from .model_102_v2 import Model as Model_102_v2

dictionary = {
    "m_B": Model,
    "m_X_102": Model_102,
    "m_X_103": Model_103,
    "m_X_102_v2": Model_102_v2
}

def get_model(model_name: str):
    return dictionary[model_name]