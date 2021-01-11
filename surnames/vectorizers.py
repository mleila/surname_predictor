import numpy as np
import pandas as pd

from surnames.vocabularies import SurnamesVocabulary, OriginVocabulary
from surnames.constants import (
    START_TOKEN,
    END_TOKEN,
    MASK_TOKEN,
    UNK_TOKEN,
    SURNAME,
    ORIGIN
)


class SurnameClassificationVectorizer:

    def __init__(
        self,
        surname_vocab: SurnamesVocabulary,
        origin_vocab: OriginVocabulary
        ):
        self.surname_vocab = surname_vocab
        self.origin_vocab = origin_vocab

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        start_token=START_TOKEN,
        end_token=END_TOKEN,
        mask_token=MASK_TOKEN,
        unknown_token=UNK_TOKEN
        ):
        surname_vocab = SurnamesVocabulary(start_token, end_token, mask_token, unknown_token)
        origin_vocab = OriginVocabulary()
        for _, row in df.iterrows():
            surname, origin = row[SURNAME], row[ORIGIN]
            for char in surname:
                surname_vocab.add_token(char)
            origin_vocab.add_token(origin)
        return cls(surname_vocab, origin_vocab)

    def vectorize_surname(self, surname, max_len):
        surname_vector = []
        surname_vector.append(self.surname_vocab.start_index)
        for char in surname:
            char_index = self.surname_vocab.lookup_token(char)
            surname_vector.append(char_index)
        surname_vector.append(self.surname_vocab.end_index)

        if len(surname_vector) < max_len:
            surname_vector += [self.surname_vocab.mask_index] * (max_len - len(surname_vector))
        else:
            surname_vector = surname_vector[:max_len-1] + [self.surname_vocab.end_index]
        return np.array(surname_vector)

    def vectorize_origin(self, token):
        return self.origin_vocab.lookup_token(token)
