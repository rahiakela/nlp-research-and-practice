
# Text Representation

## 🔖 Outline

Feature extraction is an important step for any machine learning problem. No matter
how good a modeling algorithm you use, if you feed in poor features, you will get
poor results. In computer science, this is often called **garbage in, garbage out.**

In these notebooks, we’ll address the question: how do we go about doing feature engineering for text data?
 
In other words, how do we transform a given text into numerical form so that it can be fed
into NLP and ML algorithms? 

In NLP parlance, this conversion of raw text to a suitable numerical form is called **text representation**.
 
In these notebooks, we’ll take a look at **the different(statistical) methods for text representation, or representing text as a numeric vector.**


## 🗒️ Notebooks

Set of notebooks associated with the chapter. 

1. **[One-Hot Encoding](https://github.com/rahiakela/practical-natural-language-processing/blob/master/3-text-representation/1_one_hot_encoding.ipynb)**: Here we demonstrate One-Hot encoding from the first principle as well as scikit learn's implementation on our toy corpus.

2. **[Bag of Words](https://github.com/rahiakela/practical-natural-language-processing/blob/master/3-text-representation/2_bag_of_words.ipynb)** : Here we demonstrate how to arrive at the bag of words representation for our toy corpus.
    

3. **[Bag of N Grams](https://github.com/rahiakela/practical-natural-language-processing/blob/master/3-text-representation/3_bag_of_n_gram.ipynb)**: Here we demonstrate how Bag of N-Grams work using our toy corpus.

4. **[TF-IDF](https://github.com/rahiakela/practical-natural-language-processing/blob/master/3-text-representation/4_tf_idf.ipynb)**: Here we demonstrate how to obtain the get the TF-IDF representation of a document using sklearn's TfidfVectorizer(we will be using our toy corpus). 

5. **[Pre-trained Word Embeddings](https://github.com/rahiakela/practical-natural-language-processing/blob/master/3-text-representation/5_using_pre_trained_word2vec_model.ipynb)**: Here we demonstrate how we can represent text using pre-trained word embedding models and how to use them to get representations for the full text.

6. **[Custom Word Embeddings](https://github.com/rahiakela/practical-natural-language-processing/blob/master/3-text-representation/6_training_word_embeddings.ipynb)**: Here we demonstrate how to train a custom Word Embedding model(word2vec) using gensim on both, our toy corpus and a subset of Wikipedia data.

7. **[Vector Representations via averaging](https://github.com/rahiakela/practical-natural-language-processing/blob/master/3-text-representation/7_document_vectors_using_averaging_via_spacy.ipynb)**: Here we demonstrate averaging of Document Vectors using spaCy.

8. **[Doc2Vec Model](https://github.com/rahiakela/practical-natural-language-processing/blob/master/3-text-representation/8_training_document_vectors_using_gensim.ipynb)**: Here we demonstrate how to train your own doc2vec model.

9. **[Visualizing Embeddings Using TSNE](https://github.com/rahiakela/practical-natural-language-processing/blob/master/3-text-representation/9_visualizing_embeddings_using_t_sne.ipynb)**: Here we demonstrate how we can use dimensionality reduction techniques such as TSNE to visualize embeddings.

10. **[Visualizing Embeddings using Tensorboard](https://github.com/rahiakela/practical-natural-language-processing/blob/master/3-text-representation/10_Visualizing_Embeddings_using_Tensorboard.ipynb)**: Here we demonstrate how we can visualize embeddings using Tensorboard.


## 🖼️ Figures

Color figures as used in these notebooks. 

![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-1.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-2.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-3.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-4.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-5.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-6.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-7.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-8.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-9.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-10.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-11.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-12.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-13.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-14.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-15.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-16.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-17.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-18.png)
![figure](https://github.com/practical-nlp/practical-nlp-figures/raw/master/figures/3-19.png)

