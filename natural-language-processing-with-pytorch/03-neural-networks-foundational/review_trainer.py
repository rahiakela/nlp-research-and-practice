import collections
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm_notebook

import torch
from torch import nn, optim

from review_utilities import *
from review_dataset import ReviewDataset, generate_batches
from review_classifier import ReviewClassifier
import re
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
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
)


if __name__ == '__main__':

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

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

    # The vectorizer and the model are instantiated.
    print("Instantiating the vectorizer and classifier...")
    vectorizer = dataset.get_vectorizer()
    classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
    classifier = classifier.to(args.device)

    # loss and optimizer
    print("Instantiating the loss and optimizer...")
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=1)

    ###########################################################
    ############# Step-2: The training loop ###################
    ###########################################################
    """
    More specifically, the training loop is composed of two loops: 
    1-an inner loop over minibatches in the dataset, and 
    2-an outer loop, which repeats the inner loop a number of times. 
    In the inner loop, losses are computed for each minibatch, and the optimizer is used to update the model parameters.
    """
    train_state = make_train_state(args)

    epoch_bar = tqdm_notebook(desc="split=train", total=dataset.get_num_batches(args.batch_size),
                              position=1, leave=True)

    dataset.set_split('train')
    train_bar = tqdm_notebook(desc='split=train',
                              total=dataset.get_num_batches(args.batch_size),
                              position=1,
                              leave=True)

    dataset.set_split('val')
    val_bar = tqdm_notebook(desc='split=val',
                            total=dataset.get_num_batches(args.batch_size),
                            position=1,
                            leave=True)

    try:
        for epoch_index in range(args.num_epochs):
            train_state["epoch_index"] = epoch_index

            ################################################
            ######## Iterate over training dataset #########
            ################################################

            # setup: batch generator, set loss and acc to 0, set train mode on
            dataset.set_split("train")
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            # indicate that the model is in “training mode” and the model parameters are mutable
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                # Inside each batch iteration, the optimizer’s gradients are first reset
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = classifier(x_in=batch_dict["x_data"].float())

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict["y_target"].float())
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()

                # -----------------------------------------
                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # update bar
                train_bar.set_postfix(loss=running_loss,
                                      acc=running_acc,
                                      epoch=epoch_index)
                train_bar.update()

            train_state["train_loss"].append(running_loss)
            train_state["train_acc"].append(running_acc)

            ##################################################
            #######    Iterate over val dataset ##############
            ##################################################

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split("val")
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            """
            It indicate that the model is in “validation mode”
            and makes the model parameters immutable and disables dropout
            The eval mode also disables computation of the loss
            and propagation of gradients back to the parameters
            """
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                # compute the output
                y_pred = classifier(x_in=batch_dict["x_data"].float())

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict["y_target"].float())
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict["y_target"])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                val_bar.set_postfix(loss=running_loss,
                                    acc=running_acc,
                                    epoch=epoch_index)
                val_bar.update()

            train_state["val_loss"].append(running_loss)
            train_state["val_acc"].append(running_acc)

            train_state = update_train_state(args=args, model=classifier, train_state=train_state)

            scheduler.step(train_state["val_loss"][-1])

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()

            if train_state["stop_early"]:
                break

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()
    except KeyboardInterrupt:
        print("Exiting loop")
