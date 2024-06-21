# Math Language Model

This repository contains code to create and train a language model specifically focused on mathematical content. The project includes steps for data collection, tokenization, model training, and text generation.

## Project Structure

- `data_collection.py`: Script to collect mathematical articles from Wikipedia and save them to a file.
- `tokenization.py`: Script to train a tokenizer on the collected articles.
- `model_training.py`: Script to define and train a Transformer model on the tokenized data.
- `generate_text.py`: Script to generate text using the trained model.
- `requirements.txt`: List of dependencies required for the project.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/math-language-model.git
   cd math-language-model
   ```

2. **Open GitHub Codespace**
   - Go to your GitHub repository.
   - Click on the `<> Code` button and select `Codespaces`.
   - Create a new Codespace.

3. **Install Dependencies**
   In the Codespace terminal, run:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Scripts

1. **Data Collection**
   Collect mathematical articles from Wikipedia and save them to `math_articles.txt`:
   ```bash
   python data_collection.py
   ```

2. **Tokenization**
   Train a tokenizer on the collected articles and save the tokenizer to `math_tokenizer.json`:
   ```bash
   python tokenization.py
   ```

3. **Model Training**
   Train a Transformer model on the tokenized data and save the trained model to `math_transformer_model.pth`:
   ```bash
   python model_training.py
   ```

4. **Text Generation**
   Generate text using the trained model:
   ```bash
   python generate_text.py
   ```

## Example Output

After running `generate_text.py`, you should see generated text based on the prompt provided in the script. For example:
```
The fundamental theorem of calculus states...
```

## Dependencies

- `torch`: PyTorch for model implementation and training.
- `tokenizers`: Hugging Face Tokenizers library for tokenization.
- `requests`: Library for making HTTP requests to collect data from Wikipedia.

## Acknowledgments

Special thanks to the developers of PyTorch and Hugging Face for providing powerful tools for machine learning and natural language processing.
