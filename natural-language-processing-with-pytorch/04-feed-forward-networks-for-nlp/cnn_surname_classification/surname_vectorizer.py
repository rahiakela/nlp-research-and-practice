import string
import numpy as np
from vocabulary import Vocabulary
from collections import Counter


class SurnameVectorizer(object):
    """The Vectorizer, which coordinates the Vocabularies and puts them to use"""

    def __init__(self, surname_vocab, nationality_vocab, max_surname_length):
        """
        surname_vocab (Vocabulary): maps words to integers
        nationality_vocab (Vocabulary): maps class labels to integers
        """
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab
        self._max_surname_length = max_surname_length

    def vectorize(self, surname):
        """
        Args:
            surname (str): the surname
        Returns:
            one_hot (np.ndarray): a collapsed one-hot encoding
        """
        # each character in the string to an integer and then uses that integer to construct a
        # matrix of one-hot vectors.
        one_hot_matrix_size = (len(self.surname_vocab), self._max_surname_length)
        one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)
        for position_index, character in enumerate(surname):
            character_index = self.surname_vocab.lookup_token(character)
            one_hot_matrix[character_index][position_index] = 1
        return one_hot_matrix

    @classmethod
    def from_dataframe(cls, surname_df):
        """
        Instantiate the vectorizer from the dataset dataframe
        Args:
            surname_df (pandas.DataFrame): the review dataset
        Returns:
            an instance of the SurnameVectorizer
        """
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)
        max_surname_length = 0

        for index, row in surname_df.iterrows():
            max_surname_length = max(max_surname_length, len(row.surname))
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)
        return cls(surname_vocab, nationality_vocab, max_surname_length)

    @classmethod
    def from_serializable(cls, contents):
        """
        Instantiate a SurnameVectorizer from a serializable dictionary
        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the SurnameVectorizer class
        """
        surname_vocab = Vocabulary.from_serializable(contents["surname_vocab"])
        nationality_vocab = Vocabulary.from_serializable(contents["nationality_vocab"])

        return cls(surname_vocab=surname_vocab, nationality_vocab=nationality_vocab, max_surname_length=contents["max_surname_length"])

    def to_serializable(self):
        """
        Create the serializable dictionary for caching
        Returns:
            contents (dict): the serializable dictionary
        """
        return {
            "surname_vocab": self.surname_vocab.to_serializable(),
            "nationality_vocab": self.nationality_vocab.to_serializable(),
            "max_surname_length": self._max_surname_length
        }
