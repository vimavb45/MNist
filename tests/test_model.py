from src.models.model import MyAwesomeModel

import torch

def test_model_output():
    train = torch.randn(1, 1, 28, 28)
    model = MyAwesomeModel()
    output = model(train)
    assert output.shape[1] == 10, "the model doesnt works fine"
