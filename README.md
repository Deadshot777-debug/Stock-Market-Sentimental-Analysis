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

## References
[1] Lazer, D. M., Baum, M. A., Benkler, Y., Berinsky, A. J., Greenhill, K. M., Menczer, F., ... & Zittrain, J. L. (2018). The science of fake news. Science, 359(6380), 1094-1096.

[2] Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. ACM SIGKDD explorations newsletter, 19(1), 22-36.

[3] Reis, J. C., Correia, A., Murai, F., Veloso, A., & Benevenuto, F. (2019). Supervised learning for fake news detection. IEEE Intelligent Systems, 34(2), 76-81.

[4] Gayo-Avello, D., Metaxas, P. T., Mustafaraj, E., Strohmaier, M., Schoen, H., Gloor, P., ... & Poblete, B. (2013). Predicting information credibility in time-sensitive social media. Internet Research.

[5] Jwa, H., Oh, D., Park, K., Kang, J. M., & Lim, H. (2019). exBAKE: automatic fake news detection model based on bidirectional encoder representations from transformers (bert). Applied Sciences, 9(19), 4062.

[6] Goldani, M. H., Momtazi, S., & Safabakhsh, R. (2021). Detecting fake news with capsule neural networks. Applied Soft Computing, 101, 106991.

[7] Hakak, S., Alazab, M., Khan, S., Gadekallu, T. R., Maddikunta, P. K. R., & Khan, W. Z. (2021). An ensemble machine learning approach through effective feature extraction to classify fake news. Future Generation Computer Systems, 117, 47-58.

[8] Kula, S., Choraś, M., Kozik, R., Ksieniewicz, P., & Woźniak, M. (2020, June). Sentiment analysis for fake news detection by means of neural networks. In International Conference on Computational Science (pp. 653-666). Springer, Cham.

[9] Ahmed, H., Traore, I., & Saad, S. (2018). Detecting opinion spams and fake news using text classification. Security and Privacy, 1(1), e9.

[10] Akbik, A., Bergmann, T., Blythe, D., Rasul, K., Schweter, S., & Vollgraf, R. (2019, June). FLAIR: An easy-to-use framework for state-of-the-art NLP. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations) (pp. 54-59).

[11] Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., ... & Dean, J. (2016). Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.

[12] Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural networks, 18(5-6), 602-610.

[13] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

[14] Gundapu, S., & Mamid, R. (2021). Transformer-based Automatic COVID-19 Fake News Detection System. arXiv preprint arXiv:2101.00180.

[15] Kaliyar, R. K., & Singh, N. (2019, July). Misinformation Detection on Online Social Media-A Survey. In 2019 10th International Conference on Computing, Communication and Networking Technologies (ICCCNT) (pp. 1-6). IEEE.

[16] Lee M. (2019). 進擊的 BERT：NLP 界的巨人之力與遷移學習. In https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html
