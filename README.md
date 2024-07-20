# Stock-Market-Sentiment-Analysis-NLP-GloVe-TensorFlow
This project uses Natural Language Processing (NLP) techniques to analyze the sentiment of stock market news articles, classifying them as positive or negative. It employs GloVe embeddings for feature extraction and a neural network model built with TensorFlow and Keras for sentiment classification.

# Sentiment Analysis of Stock Market News

## Introduction

In the dynamic and often volatile world of stock markets, news articles hold significant sway over investor sentiment and market movements. Recognizing the sentiment expressed in these articles is crucial for making informed investment decisions. This project leverages the power of Natural Language Processing (NLP) to analyze the sentiment of stock market news articles, categorizing them as either positive or negative.

## Objective

The main goal of this project is to develop a sentiment analysis model that can process text from news articles related to the stock market and classify the sentiment of each article. This tool aims to assist investors, financial analysts, and anyone interested in gauging market sentiment, potentially identifying trends that could lead to more strategic investment decisions.

## Technologies Used

- **Python**: The primary programming language used for the project.
- **TensorFlow and Keras**: For building and training the neural network model.
- **NLTK**: For text processing and handling natural language data.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib** and **Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning utilities like train-test split and data encoding.

## Methodology

### Data Preprocessing

Textual data often contains noise and irrelevant information. To prepare our data for analysis:
- I cleaned the text to remove any special characters, URLs, and numbers.
- Employed NLTK's stopwords to filter out uninformative words.
- Used tokenization to split text into individual words or tokens.

### Feature Extraction with GloVe Embeddings

To capture semantic meaning, I used Global Vectors for Word Representation (GloVe) for converting words into numerical vectors. This helps the model understand the context better than traditional bag-of-words models.

### Neural Network Modeling

I designed a neural network using TensorFlow and Keras, incorporating layers like Bidirectional LSTM to recognize patterns from both forward and backward context in the data, crucial for understanding the sentiment expressed in texts.

### Model Evaluation

The model's effectiveness was assessed using metrics like accuracy and ROC curves. A confusion matrix was also used to visualize the model's performance in classifying sentiments.

## Results

The model demonstrated a promising ability to classify sentiments accurately:
- Achieved an accuracy of approximately 78% on the test set.
- The ROC curve indicated good discriminative performance with an AUC of 0.78.

## Conclusion

This sentiment analysis model has shown that with effective preprocessing, feature extraction, and a robust neural network architecture, it is possible to discern the underlying sentiment in stock market news with a reasonable degree of accuracy. Going forward, this tool could be expanded to real-time news feeds for live sentiment analysis.
