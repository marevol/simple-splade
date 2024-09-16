import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SimpleSPLADE(torch.nn.Module):
    def __init__(self, model_name="xlm-roberta-base"):
        super(SimpleSPLADE, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # Get model logits from MLM
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = F.relu(output.logits)  # ReLU to ensure non-negative values

        # Apply log(1 + x) transformation for sparse representation
        sparse_repr = torch.log1p(logits).sum(dim=1)

        return sparse_repr
