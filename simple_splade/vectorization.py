import torch
import torch.nn.functional as F


class SPLADESparseVectorizer:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer

    def text_to_sparse_vector(self, text):
        """
        Converts the input text into a sparse vector.
        Returns a dictionary with token-value pairs representing the sparse vector.
        """
        # Tokenize the input text dynamically (use padding based on input length)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Model inference within no_grad context to avoid unnecessary gradient computation
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            sparse_repr = torch.log(1 + F.relu(output)).squeeze(0)  # Convert logits to sparse vector

        # Convert token IDs back to tokens (filtering out special tokens)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        sparse_vector = {
            token: sparse_repr[idx].item()
            for idx, token in enumerate(tokens)
            if sparse_repr[idx] > 0 and token not in ["[PAD]", "[CLS]", "[SEP]"]
        }

        return sparse_vector
