import torch

from simple_splade.model import SimpleSPLADE


def test_splade_forward():
    model = SimpleSPLADE()
    input_ids = torch.tensor([[101, 2054, 2003, 1996, 2087, 2518, 102]])  # Example input
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])

    output = model(input_ids, attention_mask)
    assert output is not None
    assert output.shape == (1, 30522)  # Example for BERT vocab size
