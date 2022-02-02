import re

import torch

from review_utilities import *
from review_dataset import ReviewDataset, generate_batches
from review_classifier import ReviewClassifier
import os
from argparse import Namespace

# we make use of an args object to centrally coordinate all decision points
args = Namespace(
    frequency_cutoff=25,
    model_state_file="model.pth",
    review_csv="D:\\ml-datasets\\yelp-review-dataset\\reviews_with_splits_full.csv",
    save_dir="D:\\ml-datasets\\yelp-review-dataset\\model_storage\\",
    vectorizer_file='vectorizer.json',
    # No Model hyper parameters
    # Training hyper parameters
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    seed=1337,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=False,
    expand_filepaths_to_save_dir=False,
    reload_from_files=True,
)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    """
    Predict the rating of a review
    Args:
        review (str): the text of the review
        classifier (ReviewClassifier): the trained model
        vectorizer (ReviewVectorizer): the corresponding vectorizer
        decision_threshold (float): The numerical boundary which separates the rating classes
    """
    review = preprocess_text(review)

    vectorized_review = torch.tensor(vectorizer.vectorize(review))
    result = classifier(vectorized_review.view(1, -1))

    probability_value = torch.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0
    return vectorizer.rating_vocab.lookup_index(index)


if __name__ == '__main__':

    # set vectorizer and model state file path
    args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
    args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

    if args.expand_filepaths_to_save_dir:
        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))

    if not torch.cuda.is_available():
        args.cuda = False

    print("Using CUDA: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Set seed for reproducibility
    set_seed_everywhere(args.seed, args.cuda)
    # handle dirs
    handle_dirs(args.save_dir)

    ###########################################################
    ############# Step-1: Initializations #####################
    ###########################################################
    if args.reload_from_files:
        # training from a checkpoint
        print("Loading dataset and vectorizer...")
        dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv, args.vectorizer_file)
    else:
        print("Loading dataset and creating vectorizer...")
        # create dataset and vectorizer
        dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
        dataset.save_vectorizer(args.vectorizer_file)

    train_state = make_train_state(args)

    # compute the loss & accuracy on the test set using the best available model
    print("Instantiating the vectorizer and classifier...")
    vectorizer = dataset.get_vectorizer()
    classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
    classifier.load_state_dict(torch.load(train_state["model_filename"]))
    classifier = classifier.cpu()

    # we do inference on new data and make qualitative judgments about whether the model is working
    test_review = "this is a pretty awesome book"

    prediction = predict_rating(test_review, classifier, vectorizer, decision_threshold=0.5)
    print("{} -> {}".format(test_review, prediction))

    test_review = "Today, I am feeling awesome"

    prediction = predict_rating(test_review, classifier, vectorizer, decision_threshold=0.8)
    print("{} -> {}".format(test_review, prediction))

    test_review = "I do not like such a boring book"

    prediction = predict_rating(test_review, classifier, vectorizer, decision_threshold=0.8)
    print("{} -> {}".format(test_review, prediction))
