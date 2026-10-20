# SentimentAI - Movie Review Sentiment Analysis

🌐 **Live Demo:** https://sentimentai-mzey.onrender.com


# SentimentAI - Movie Review Sentiment Analysis

SentimentAI is a web application that predicts whether a movie review expresses positive or negative sentiment. It uses an LSTM (Long Short-Term Memory) neural network built with TensorFlow and Keras, with a Streamlit interface for interactive predictions.

## Features

- Predicts positive or negative sentiment from movie reviews.
- Displays the sentiment prediction score.
- Includes sample reviews for testing.
- Maintains prediction history during the current session.
- Displays model evaluation metrics and visualizations.
- Provides a simple and interactive interface.

## Technologies Used

- Python
- TensorFlow and Keras
- LSTM Neural Network
- Streamlit
- NumPy
- scikit-learn
- Matplotlib

## Dataset

The model was trained using the IMDb movie review dataset provided through Keras.

- Training reviews: 25,000
- Testing reviews: 25,000
- Vocabulary size: 10,000
- Maximum sequence length: 200 tokens
- Training epochs: 3

## Model Architecture

The model consists of three main layers:

1. **Embedding Layer:** Converts word IDs into numerical vector representations.
2. **LSTM Layer:** Processes the review sequence to learn contextual patterns.
3. **Dense Output Layer:** Uses sigmoid activation to produce a sentiment score.

A threshold of 0.5 is used to classify a review as positive or negative.

## Model Performance

The following results were obtained on the held-out IMDb test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 84.05% |
| Precision | 83.42% |
| Recall | 84.98% |
| F1-Score | 84.20% |

Performance may vary on new reviews. The model can struggle with sarcasm, negation, and mixed opinions.

## Project Structure

```text
SentimentAI/
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── models/
    ├── sentiment_lstm.keras
    ├── imdb_word_index.npz
    ├── training_accuracy.png
    └── confusion_matrix.png
```

## Installation and Setup

### 1. Clone the repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/SentimentAI.git
cd SentimentAI
```

### 2. Create a virtual environment

Python 3.12 is recommended for the environment used during development.

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

## Train the Model

To train the model again, run:

```bash
python train.py
```

The training script downloads the IMDb dataset when needed, trains the model, and saves the generated model artifacts and evaluation plots in the `models/` directory.

## How It Works

1. The user enters a movie review.
2. The review is converted into numerical word IDs.
3. The sequence is padded or truncated to 200 tokens.
4. The trained LSTM model processes the sequence.
5. The model generates a sentiment score.
6. The application displays the predicted sentiment.

## Limitations

- Predictions are not always correct.
- Sarcasm, negation, and mixed sentiment can be difficult to classify.
- The model is trained on movie reviews and may perform differently on other types of text.
- The prediction score should not be interpreted as a calibrated probability of correctness.

## Future Improvements

- Improve handling of negation and sarcasm.
- Compare the LSTM model with alternative text-classification approaches.
- Add more detailed error analysis.
- Add automated tests for text preprocessing.
- Deploy the application online.

## Author

**Varun Kumar**

MCA - Data Science
varun-kumar.in