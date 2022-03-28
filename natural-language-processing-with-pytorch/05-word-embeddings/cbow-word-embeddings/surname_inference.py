from cbow_utilities import *
from cbow_dataset import SurnameDataset
from cbow_classifier import SurnameClassifier
import os
from argparse import Namespace

# we make use of an args object to centrally coordinate all decision points
args = Namespace(
    model_state_file="model.pth",
    surname_csv="D:\\ml-datasets\\surnames-dataset\\surnames_with_splits.csv",
    save_dir="D:\\ml-datasets\\surnames-dataset\\surname_mlp\\",
    vectorizer_file='vectorizer.json',
    # Model hyper parameters
    hidden_dim=300,
    num_channels=256,
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


def predict_nationality(surname, classifier, vectorizer, k=5):
    vectorized_surname = vectorizer.vectorize(surname)
    vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(0)
    result = classifier(vectorized_surname, apply_softmax=True)

    probability_values, indices = result.max(dim=1)
    index = indices.item()

    predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)
    probability_value = probability_values.item()

    return {"nationality": predicted_nationality, "probability": probability_value}


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
        dataset = SurnameDataset.load_dataset_and_load_vectorizer(args.surname_csv, args.vectorizer_file)
    else:
        print("Loading dataset and creating vectorizer...")
        # create dataset and vectorizer
        dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
        dataset.save_vectorizer(args.vectorizer_file)

    train_state = make_train_state(args)

    # compute the loss & accuracy on the test set using the best available model
    print("Instantiating the vectorizer and classifier...")
    vectorizer = dataset.get_vectorizer()
    vectorizer.nationality_vocab.lookup_index(8)

    classifier = SurnameClassifier(initial_num_channels=len(vectorizer.surname_vocab),
                                   num_classes=len(vectorizer.nationality_vocab),
                                   num_channels=args.num_channels)
    classifier.load_state_dict(torch.load(train_state["model_filename"]))
    classifier = classifier.to(args.device)
    classifier = classifier.cpu()

    # It is often useful to look at more than just the best prediction
    while True:
        new_surname = input("Enter a surname to classify: ")

        prediction = predict_nationality(new_surname, classifier, vectorizer)
        print("==============================")
        print("{} -> {} (p={:0.2f})".format(new_surname,
                                prediction["nationality"],
                                prediction["probability"]))
