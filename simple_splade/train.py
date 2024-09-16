import logging

import torch
import torch.nn.functional as F


def splade_ranking_loss(query_output, positive_output, negative_outputs):
    """
    Implements the SPLADE ranking loss function using log-sum-exp for numerical stability.

    Args:
        query_output (Tensor): Query embedding.
        positive_output (Tensor): Positive document embedding.
        negative_outputs (Tensor): Negative document embeddings.

    Returns:
        Tensor: Ranking loss.
    """
    # Compute cosine similarities for the positive and negative documents
    positive_similarity = F.cosine_similarity(query_output, positive_output, dim=-1)
    negative_similarities = F.cosine_similarity(query_output.unsqueeze(1), negative_outputs, dim=-1)

    # Numerically stable logsumexp for ranking loss
    loss = -torch.log_softmax(torch.cat([positive_similarity.unsqueeze(1), negative_similarities], dim=1), dim=1)[:, 0]

    return loss.mean()


def train_with_ranking_loss(model, dataloader, optimizer, num_epochs=3, device="cpu"):
    """
    Trains the SPLADE model using the ranking loss function.

    Args:
        model (SimpleSPLADE): SPLADE model.
        dataloader (DataLoader): Dataloader for triplets of query, positive, and negative samples.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to train on (CPU or GPU).

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Move batch to device
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            positive_input_ids = batch["positive_input_ids"].to(device)
            positive_attention_mask = batch["positive_attention_mask"].to(device)
            negative_input_ids = batch["negative_input_ids"].to(device)
            negative_attention_mask = batch["negative_attention_mask"].to(device)

            # Forward pass
            query_output = model(query_input_ids, query_attention_mask)
            positive_output = model(positive_input_ids, positive_attention_mask)
            negative_outputs = model(negative_input_ids, negative_attention_mask)

            # Compute ranking loss
            loss = splade_ranking_loss(query_output, positive_output, negative_outputs)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")
