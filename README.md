# NLP Project: BBC News Language Modeling

This repository contains code and resources for training and evaluating language models on the BBC News Summary dataset. The project explores both a custom LSTM-based language model and a pre-trained GPT-2 model for text generation and perplexity evaluation.

## Project Structure

- `nlp_project_script.py` — Main Python script containing all data processing, model training, and evaluation code.
- `Report.pdf` — Project report (see for detailed methodology, results, and discussion).

## Dataset

The project uses the BBC News Summary dataset, which contains news articles categorized by topic. The dataset is loaded and preprocessed in the script. Please ensure the dataset is available at the specified path or update the path in the script accordingly.

## Main Features

- **Data Preprocessing:**
  - Loads and cleans news articles.
  - Tokenizes and encodes text for model input.
  - Splits data into training and validation sets.

- **LSTM Language Model:**
  - Custom PyTorch implementation.
  - Trains on BBC news articles.
  - Generates text based on user prompts.
  - Reports training and validation perplexity.

- **GPT-2 Evaluation:**
  - Uses HuggingFace Transformers to load GPT-2.
  - Evaluates GPT-2 perplexity on validation data.
  - Generates text from prompts using GPT-2.

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- tqdm
- transformers (for GPT-2 evaluation)

Install dependencies with:
```bash
pip install torch scikit-learn tqdm transformers
```

## Usage

1. **Prepare the Dataset:**
   - Download and extract the BBC News Summary dataset.
   - Update the `DATA_DIR` variable in the script if needed.

2. **Run the Script:**
   ```bash
   python nlp_project_script.py
   ```
   - The script will train the LSTM model, evaluate perplexity, and generate sample outputs.
   - GPT-2 evaluation and generation are included (requires internet for model download).

3. **Results:**
   - Generated outputs and evaluation results are saved to `generated_outputs.txt` and `results.txt`.
   - See `Report.pdf` for detailed analysis and findings.

## Project Report

For a comprehensive explanation of the methodology, experiments, and results, please refer to `Report.pdf` in this repository.

## License

This project is for educational purposes. Please check dataset and model licenses for any usage restrictions.



*Created February 2026*

# Fake-News-Prediction-using-NLP

