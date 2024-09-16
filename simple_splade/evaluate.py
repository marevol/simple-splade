import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluate_with_ranking_loss(model, dataloader, device="cpu"):
    """
    Evaluate the SPLADE model using ranking loss on a dataset.

    Args:
        model (SimpleSPLADE): The trained model to evaluate.
        dataloader (DataLoader): Dataloader containing the test samples.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: Average ranking loss on the evaluation set.
        float: Accuracy of the model based on ranking.
    """
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            positive_input_ids = batch["positive_input_ids"].to(device)
            positive_attention_mask = batch["positive_attention_mask"].to(device)
            negative_input_ids = batch["negative_input_ids"].to(device)
            negative_attention_mask = batch["negative_attention_mask"].to(device)

            # Model output
            query_output = model(query_input_ids, query_attention_mask)
            positive_output = model(positive_input_ids, positive_attention_mask)
            negative_output = model(negative_input_ids, negative_attention_mask)

            # Compute cosine similarity
            positive_similarity = F.cosine_similarity(query_output, positive_output, dim=-1)
            negative_similarity = F.cosine_similarity(query_output, negative_output, dim=-1)

            # Calculate ranking loss using logsumexp directly for stability
            similarities = torch.cat([positive_similarity.unsqueeze(1), negative_similarity.unsqueeze(1)], dim=1)
            loss = -F.log_softmax(similarities, dim=1)[:, 0].mean()
            total_loss += loss.item()

            # Accuracy based on ranking
            correct += (positive_similarity > negative_similarity).sum().item()
            total += len(positive_similarity)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Evaluation - Average Ranking Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
