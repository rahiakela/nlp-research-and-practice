# Text Classification

## 🔖 Outline

Text classification is the task of assigning one or more categories to a given piece of text from a larger set of possible categories.

Text classification is a special instance of the
classification problem, where the input data point(s) is text and the goal is to categorize
the piece of text into one or more buckets (called a class) from a set of predefined
buckets (classes). The “text” can be of arbitrary length: a character, a word, a
sentence, a paragraph, or a full document.

Any supervised classification approach, including text classification, can be further
distinguished into three types based on the number of categories involved: binary,
multiclass, and multilabel classification. If the number of classes is two, it’s called
binary classification. If the number of classes is more than two, it’s referred to as multiclass
classification.

Thus, classifying an email as spam or not-spam is an example of
binary classification setting. Classifying the sentiment of a customer review as negative,
neutral, or positive is an example of multiclass classification. In both binary and
multiclass settings, each document belongs to exactly one class from C, where C is the
set of all possible classes. In multilabel classification, a document can have one or
more labels/classes attached to it.

Text classification is sometimes also referred to as topic classification, text categorization,
or document categorization. For the rest of this book, we’ll stick to the term “text
classification.” Note that topic classification is different from topic detection, which
refers to the problem of uncovering or extracting “topics” from texts.



## 🗒️ Notebooks

Set of notebooks associated with the chapter. 

1. **[One Pipeline Many Classifiers](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/1_statistical_text_classification_with_naive_bayes_logistic_regression_and_svm.ipynb)**: Here we demonstrate text classification using various algorithms such as Naive Bayes, Logistic Regression, and Support Vector Machines.

2. **[Doc2Vec for Text Classification](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/2_word_embeddings.ipynb)**: Here we demonstrate how to train your own Doc2Vec embedding and use it for text classification.

3. **[Word2Vec for Text Classification](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/3_subword_embeddings_and_fast_text.ipynb)**: Here we demonstrate how to use a pre-trained Word2Vec model for text classification.

4. **[FastText for Text Classification](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/4_document_embeddings.ipynb)**: Here we demonstrate how to use the fasttext library for text classification.

5. **[NNs for Text Classification](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/5_deep_learning_for_text_classification.ipynb)**: Here we demonstrate text classification using pre-trained and custom word embeddings with various Deep Learning Models. 

6. **[BERT: Text Classification](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/6_BERT_text_classification_with_large_pre_trained_language_models.ipynb)**: Here we demonstrate how we train and fine-tune pytorch pre-trained BERT on IMDB reviews to predict their sentiment using HuggingFace Transformers library.

7. **[BERT: Text CLassification using Ktrain](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/7_BERT_sentiment_classification_with_large_pre_trained_model_using_ktrain.ipynb)**: Here we demonstrate how we can use BERT to predict the sentiment of movie reviews using the ktrain library.

8. **[LIME-1](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/8_interpreting_text_classification_models_using_lime.ipynb)**: Here we demonstrate how to interpret the predictions of a logistic regression model using LIME.

9. **[LIME-2](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/9_interpreting_rnn_models_using_lime.ipynb)**: Here we demonstrate how to interpret predictions of an RNN model using LIME.

10. **[SHAP](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/10_explaining_logistic_regression_classifier_and_RNN_model_using_shap.ipynb)**: Here we demonstrate how to interpret ML and DL text classification models using SHAP.

11. **[Spam Classification](https://github.com/rahiakela/practical-natural-language-processing/blob/master/4-text-classification/11_sms_spam_classification.ipynb)**: Here we demonstrate how to classify a text message as SPAM or HAM using pre-trained models from the fastai library. 


