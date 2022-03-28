import torch
from torch import nn, optim

from cbow_utilities import *
from cbow_dataset import CBOWDataset, generate_batches
from cbow_classifier import CBOWClassifier
import os
from argparse import Namespace

# we make use of an args object to centrally coordinate all decision points
args = Namespace(
    model_state_file="model.pth",
    cbow_csv="D:\\ml-datasets\\frankenstein-dataset\\frankenstein_with_splits.csv",
    save_dir="D:\\ml-datasets\\frankenstein-dataset\\model_storage\\",
    vectorizer_file='vectorizer.json',
    # Model hyper parameters
    embedding_size=50,
    # Training hyper parameters
    batch_size=32,
    early_stopping_criteria=5,
    dropout_p=0.5,
    learning_rate=0.0001,
    num_epochs=100,
    seed=1337,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=False,
    expand_filepaths_to_save_dir=False,
    reload_from_files=True,
)

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
        dataset = CBOWDataset.load_dataset_and_load_vectorizer(args.cbow_csv, args.vectorizer_file)
    else:
        print("Loading dataset and creating vectorizer...")
        # create dataset and vectorizer
        dataset = CBOWDataset.load_dataset_and_make_vectorizer(args.cbow_csv)
        dataset.save_vectorizer(args.vectorizer_file)

    train_state = make_train_state(args)

    # compute the loss & accuracy on the test set using the best available model
    print("Instantiating the vectorizer and classifier...")
    vectorizer = dataset.get_vectorizer()
    classifier = CBOWClassifier(vocabulary_size=len(vectorizer.cbow_vocab), embedding_size=args.embedding_size)
    classifier.load_state_dict(torch.load(train_state["model_filename"]))
    classifier = classifier.to(args.device)

    # loss and optimizer
    print("Instantiating the loss and optimizer...")
    loss_func = nn.CrossEntropyLoss()

    ###########################################################
    ############# Step-2: The testing loop ###################
    ###########################################################
    """
    To evaluate the data on the held-out test set, the code is exactly the same as the validation
    loop in the training routine.
    The difference between the two partitions of the dataset comes from the fact 
    that the test set should be run as little as possible.
    """
    dataset.set_split('test')
    print(f"Batch size: {args.batch_size}")
    batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    """
    It indicate that the model is in “validation mode” and makes the model parameters immutable 
    and disables dropout The eval mode also disables computation of the loss
    and propagation of gradients back to the parameters
    """
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = classifier(x_in=batch_dict["x_data"])

        # compute the loss
        loss = loss_func(y_pred, batch_dict["y_target"])
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_batch = compute_accuracy(y_pred, batch_dict["y_target"])
        running_acc += (acc_batch - running_acc) / (batch_index + 1)

    train_state["test_loss"] = running_loss
    train_state["test_acc"] = running_acc

    print("Test loss: {:.3f}".format(train_state["test_loss"]))
    print("Test acc: {:.3f}".format(train_state["test_acc"]))
