# Simple SPLADE

This project implements a simplified version of SPLADE (Sparse Lexical and Expansion Model), leveraging sparse representations for efficient retrieval using masked language models (MLMs), such as `xlm-roberta-base`. The project is designed for training, sparse vectorization, and retrieval tasks.

## Features

- **Training**: Train the SPLADE model using masked language models with a ranking loss.
- **Sparse Vectorization**: Convert text inputs into sparse representations (word-value pairs).
- **Multilingual Support**: Supports multilingual text with pre-trained models like `xlm-roberta-base`.

## Installation

### Requirements

- Python 3.10+
- Poetry
- PyTorch
- Transformers (Hugging Face)
- Pandas

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/marevol/simple-splade.git
   cd simple-splade
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

   This will create a virtual environment and install all the necessary dependencies listed in `pyproject.toml`.

3. Activate the virtual environment created by Poetry:
   ```bash
   poetry shell
   ```

## Data Preparation

This project relies on the **Amazon ESCI dataset** for training the model. You need to download the dataset and place it in the correct directory.

1. Download the dataset:
   - Download the **shopping_queries_dataset_products.parquet** and **shopping_queries_dataset_examples.parquet** files from the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data).

2. Place the downloaded files in the `downloads` directory within your project folder:
   ```bash
   ./downloads/shopping_queries_dataset_products.parquet
   ./downloads/shopping_queries_dataset_examples.parquet
   ```

3. The `main.py` script is set to load the dataset from the `downloads` directory by default. If you wish to place the files elsewhere, modify the paths in the script accordingly.

## Usage

### Running the Sample Script

The `main.py` script demonstrates how to use the **Amazon ESCI dataset** to train the SPLADE model, save it, and then use the trained model to convert text into sparse vectors for retrieval.

To run the sample execution with the Amazon ESCI dataset:

```bash
poetry run python main.py
```

This script performs the following steps:

1. **Training**: It loads the product titles from the Amazon ESCI dataset, trains the SPLADE model on the titles, and saves the trained model.
2. **Sparse Vectorization**: After training, the model is used to convert a sample text into a sparse vector representation.
3. **Retrieval**: It demonstrates retrieval using the generated sparse vectors from dummy data.

You can modify the script or dataset paths as needed.

### File Structure

- `main.py`: The main entry point for running the sample execution using the Amazon ESCI dataset.
- `simple_splade/vectorization.py`: Contains the `SPLADESparseVectorizer` class for converting text into sparse vectors.
- `simple_splade/model.py`: Defines the `SimpleSPLADE` model architecture.
- `simple_splade/train.py`: Handles the training process for the SPLADE model.
- `simple_splade/evaluate.py`: Contains functions for evaluating the model using ranking loss.

### Output

Once the script completes, the following will happen:

1. A trained model will be saved in the `splade_model` directory.
2. Sparse vector representations for the example text will be printed in the console.
3. Retrieval results will be shown for a dummy query against indexed dummy documents.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.

