# Sentiment Analysis on Customer Tweets (Binary Text Classification)

This project performs binary sentiment classification on customer tweets related to tech companies. It applies advanced Natural Language Processing (NLP) techniques to detect negative sentiment using both traditional machine learning (Logistic Regression, XGBoost) and deep learning (LSTM, CNN). Ensemble methods and various embeddings (TF-IDF, GloVe) help improve classification accuracy.

## Features

- **Data preprocessing:** Cleans and prepares tweets for analysis (removes URLs, mentions, hashtags, punctuation, stopwords, applies stemming).
- **Visualization:** Includes exploratory data analysis of tweet lengths and label distributions.
- **Tokenization & Padding:** Converts processed text into sequences for neural network input.
- **Deep Learning Model:** Uses a Bidirectional LSTM neural network with embedding, dropout, and dense layers for binary classification.
- **Validation:** Splits the training set for model validation and visualizes accuracy/loss over epochs.
- **Submission:** Generates a CSV with predictions for the test set.

## Workflow

1. **Data Loading**
   - Reads `train.csv` and `test.csv` containing tweet text and labels.

2. **Exploratory Data Analysis**
   - Analyzes tweet lengths, missing values, and label distribution.
   - Visualizes these statistics using matplotlib and seaborn.

3. **Text Preprocessing**
   - Cleans text: removes URLs, mentions, hashtags, punctuation, stopwords, and applies stemming.
   - Shows the effect of cleaning on sample tweets.
   - Identifies common words before and after cleaning.

4. **Tokenization & Padding**
   - Converts text to integer sequences using Keras Tokenizer.
   - Pads sequences to a fixed length for neural network input.

5. **Model Building**
   - Constructs a Bidirectional LSTM model using TensorFlow/Keras.
   - Embedding layer maps words to vectors, followed by LSTM, dropout, and dense layers.
   - Compiles the model with binary cross-entropy loss and Adam optimizer.

6. **Model Training & Validation**
   - Trains the model and validates it on a held-out subset.
   - Plots training/validation accuracy and loss.

7. **Prediction & Submission**
   - Predicts sentiment labels for the test set.
   - Outputs results as `submission.csv`.

## File Structure

- `sentiment_analysis_on_customer_tweets_binary_text_classification.py`: Main Python script containing the end-to-end workflow.
- `train_2kmZucJ.csv`: Training data (tweets, labels).
- `test_12QyDcx.csv`: Test data for predictions.
- `submission (1).csv`: Example submission file.
- `LICENSE`: License information.

## Dependencies

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- nltk
- scikit-learn
- tensorflow, keras

Install dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow
```

**Note:** NLTK stopwords are downloaded within the script.

## Usage

1. Place your datasets (`train.csv`, `test.csv`) in the working directory.
2. Run the main script:

```bash
python sentiment_analysis_on_customer_tweets_binary_text_classification.py
```

3. The model will train, evaluate, and output predictions to `submission.csv`.

## References

- [Original Colab Notebook](https://colab.research.google.com/drive/1QU3X_WAl2rnyhn3u83vqHd-hju0lKaYI)
- [Project Repository](https://github.com/manola1109/Sentiment-Analysis-on-Customer-Tweets-Binary-Text-Classification)

## License

See [LICENSE](LICENSE) for details.

