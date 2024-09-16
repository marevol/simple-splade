import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from simple_splade.evaluate import evaluate_with_ranking_loss
from simple_splade.model import SimpleSPLADE
from simple_splade.train import train_with_ranking_loss
from simple_splade.vectorization import SPLADESparseVectorizer


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("train_splade.log")],
    )


def drop_insufficient_data(df):
    id_df = df[["query_id", "exact"]]
    id_df.loc[:, ["total"]] = 1
    id_df = id_df.groupby("query_id").sum().reset_index()
    id_df = id_df[id_df.exact > 0]
    id_df = id_df[id_df.exact != id_df.total]
    return pd.merge(id_df[["query_id"]], df, how="left", on="query_id")


def load_data():
    product_df = pd.read_parquet("downloads/shopping_queries_dataset_products.parquet")
    example_df = pd.read_parquet("downloads/shopping_queries_dataset_examples.parquet")
    df = pd.merge(
        example_df[["example_id", "query_id", "product_id", "query", "esci_label", "split"]],
        product_df[["product_id", "product_title"]],
        how="left",
        on="product_id",
    )[["example_id", "query_id", "query", "product_title", "esci_label", "split"]]
    df["exact"] = df.esci_label.apply(lambda x: 1 if x == "E" else 0)
    train_df = drop_insufficient_data(
        df[df.split == "train"][["example_id", "query_id", "query", "product_title", "exact"]]
    )
    test_df = drop_insufficient_data(
        df[df.split == "test"][["example_id", "query_id", "query", "product_title", "exact"]]
    )
    return train_df, test_df


class QueryDocumentTripletDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128, size=0):
        """
        Dataset for query, positive, and negative triplet samples.
        Args:
            df (DataFrame): DataFrame containing query and document information.
            tokenizer (AutoTokenizer): Tokenizer to encode the queries and documents.
            max_length (int): Maximum length for tokenization.
            size (int): Subset size (if > 0, limits the dataset size).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.queries = df.groupby("query_id")
        self.query_ids = list(self.queries.groups.keys())
        if size > 0:
            self.query_ids = self.query_ids[:size]

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        query_group = self.queries.get_group(self.query_ids[idx])
        query = query_group.iloc[0]["query"]

        # Sample positive document
        positive_sample = query_group[query_group["exact"] == 1]
        if positive_sample.empty:
            raise ValueError(f"No positive samples for query_id: {self.query_ids[idx]}")
        positive_doc = positive_sample.sample(1).iloc[0]["product_title"]

        # Sample negative document
        negative_sample = query_group[query_group["exact"] == 0]
        if negative_sample.empty:
            raise ValueError(f"No negative samples for query_id: {self.query_ids[idx]}")
        negative_doc = negative_sample.sample(1).iloc[0]["product_title"]

        # Tokenize query, positive, and negative documents
        query_encoding = self.tokenizer(
            query, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        positive_encoding = self.tokenizer(
            positive_doc, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        negative_encoding = self.tokenizer(
            negative_doc, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        return {
            "query_input_ids": query_encoding["input_ids"].squeeze(0),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoding["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoding["attention_mask"].squeeze(0),
            "negative_input_ids": negative_encoding["input_ids"].squeeze(0),
            "negative_attention_mask": negative_encoding["attention_mask"].squeeze(0),
        }


def save_model(logger, model, optimizer=None, save_directory="splade_model"):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    model.model.save_pretrained(save_directory)
    model.tokenizer.save_pretrained(save_directory)

    if optimizer:
        optimizer_path = os.path.join(save_directory, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_path)

    logger.info(f"Model and optimizer saved to {save_directory}")


def load_model(save_directory="splade_model"):
    model = SimpleSPLADE(model_name=save_directory)
    print(f"Model loaded from {save_directory}")
    return model


def vectorize(logger):
    # Reload model
    logger.info("Reloading model for testing...")
    model = load_model()

    vectorizer = SPLADESparseVectorizer(model)

    # Example input text
    input_texts = [
        "TOPTIE Men's Long Sleeve Coverall, Snap and Zip-Front Coverall-Gray-L Regular",
        "INSE Aspiradora Escoba con Cable, 3 En 1 Vertical y de Mano, Aspirador con Cable Succión Poderosa de 18Kpa, 600W, 1L, Hepa Filtro Lavable, 3 Cepillos Ajustable, para Pelo de Mascota Suelo Duro (Rojo)",
        "CamelBak CLASSIC Hydration Pack, 85oz",
        "Coolaroo Replacement Cover, The Original Elevated Pet Bed by Coolaroo, Large, Brunswick Green",
        "ULAK iPhone 6 Plus Case, iPhone 6S Plus Case, Slim Dual Layer Soft Silicone and Hard Back Cover Anti Scratches Bumper Protective Case for Apple iPhone 6 Plus / 6S Plus 5.5 inch - Rose Gold",
        "CUK Mantis Custom Gamer PC (Liquid Cooled AMD Ryzen 9 5950X, 32GB DDR4 RAM, 512GB NVMe SSD + 2TB HDD, NVIDIA GeForce RTX 3070 8GB, Windows 11 Home) Tower Gaming Desktop Computer",
        "[オリエント] ORIENT SUY04005A0 (グレー) 海外モデル 日本製 クオーツ 腕時計 レディース 《並行輸入品》",
        "MIRTUX Kit de repuestos Conga Excellence y 990 Excellence. Pack de Accesorios de Recambio para Robots aspiradora Conga con Cepillo Lateral, Rodillo Central, filtros, prefiltro y mopa.",
        "Nip + Fab Glycolic Fix Night Pads Extreme, 2.7 Oz, 60 Count",
        "FAIRYGEM You're My Person Necklaces,Sterling Silver Interlocking Circles, Friendship Gifts for Women Friends, Birthday",
    ]

    for input_text in input_texts:
        # Convert input text to sparse vector
        sparse_vector = vectorizer.text_to_sparse_vector(input_text)

        # Log the sparse vector results
        logger.info(input_text)
        for token, value in sparse_vector.items():
            logger.info(f"{token}: {value}")


def train(logger, train_df, test_df=None, num_train=4000, num_test=800):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = SimpleSPLADE(model_name="xlm-roberta-base").to(device)

    logger.info("Preparing dataset and dataloader...")
    train_dataset = QueryDocumentTripletDataset(train_df, tokenizer, size=num_train)
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    logger.info("Starting training with ranking loss...")
    train_with_ranking_loss(model, dataloader, optimizer, num_epochs=2, device=device)
    logger.info("Training completed.")

    save_model(logger, model, optimizer)

    if test_df is not None:
        test_dataset = QueryDocumentTripletDataset(test_df, tokenizer, size=num_test)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        logger.info("Evaluating the model on the test set with ranking loss...")
        evaluate_with_ranking_loss(model, test_dataloader, device=device)


if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger(__name__)

    logger.info("Loading data from Amazon ESCI dataset...")
    train_df, test_df = load_data()
    logger.info(f"Train data: {len(train_df)}, Test data: {len(test_df)}")

    logger.info("Starting SPLADE training with Amazon ESCI dataset...")
    train(logger, train_df, test_df)

    vectorize(logger)
