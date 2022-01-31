import collections
import numpy as np
import pandas as pd
import re
import os
from argparse import Namespace

args = Namespace(
    base_dataset_dir="D:\\ml-datasets\\yelp-review-dataset\\",
    raw_train_dataset_csv="raw_train.csv",
    raw_test_dataset_csv="raw_test.csv",
    train_proportion=0.7,
    val_proportion=0.3,
    output_munged_csv="reviews_with_splits_full.csv",
    seed=1337
)


def preprocess_text(text):
    if type(text) == float:
        print(text)
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


if __name__ == '__main__':
    # Read raw data
    train_reviews = pd.read_csv(os.path.join(args.base_dataset_dir + args.raw_train_dataset_csv), header=None, names=['rating', 'review'])
    train_reviews = train_reviews[~pd.isnull(train_reviews.review)]
    test_reviews = pd.read_csv(os.path.join(args.base_dataset_dir + args.raw_test_dataset_csv), header=None, names=['rating', 'review'])
    test_reviews = test_reviews[~pd.isnull(test_reviews.review)]

    # Splitting train by rating
    by_rating = collections.defaultdict(list)
    for _, row in train_reviews.iterrows():
        by_rating[row.rating].append(row.to_dict())  # Create dict

    # Create split data
    final_list = []
    np.random.seed(args.seed)

    for _, item_list in sorted(by_rating.items()):
        np.random.shuffle(item_list)

        n_total = len(item_list)
        n_train = int(args.train_proportion * n_total)
        n_val = int(args.val_proportion * n_total)

        # Give data point a split attribute
        for item in item_list[:n_train]:
            item["split"] = "train"

        for item in item_list[n_train: n_train + n_val]:
            item["split"] = "val"

        # Add to final list
        final_list.extend(item_list)

    # now do the same for test set
    for _, row in test_reviews.iterrows():
        row_dict = row.to_dict()
        row_dict["split"] = "test"
        final_list.append(row_dict)

    # Write split data to file
    final_reviews = pd.DataFrame(final_list)

    # Preprocess the reviews
    final_reviews.review = final_reviews.review.apply(preprocess_text)

    # now label the rating
    final_reviews["rating"] = final_reviews.rating.apply({1: "negative", 2: "positive"}.get)

    # finally, save the preprocessed dataset
    final_reviews.to_csv(os.path.join(args.base_dataset_dir, args.output_munged_csv), index=False)
    print(f"Dataset preprocessing is done.......")
