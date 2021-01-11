import torch
import pandas as pd

from surnames.vectorizers import SurnameClassificationVectorizer
from surnames.constants import (
    TRAIN,
    VALID,
    TEST,
    SURNAME,
    ORIGIN,
    X_DATA,
    Y_TARGET
    )


class SurnameClassificationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        surname_vectorizer: SurnameClassificationVectorizer,
        max_surname_length: int=10
        ):
        self.df = df
        self.surname_vectorizer = surname_vectorizer
        self.max_surname_length = max_surname_length
        self.set_split()

    @classmethod
    def from_dataframe(cls, df):
        surname_vectorizer = SurnameClassificationVectorizer.from_dataframe(df)
        return cls(df, surname_vectorizer)

    def set_split(self, split: str=TRAIN):
        assert split in [TRAIN, VALID, TEST], f'Split must be either {TRAIN}, {VALID}, or {TEST}'
        self._target_df = self.df.query(f'split=="{split}"')

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        surname, origin = row[SURNAME], row[ORIGIN]
        surname_vector = self.surname_vectorizer.vectorize_surname(surname, self.max_surname_length)
        origin_vector = self.surname_vectorizer.vectorize_origin(origin)
        return {
            X_DATA: surname_vector,
            Y_TARGET: origin_vector
        }

    def __len__(self):
        return len(self._target_df)


def generate_batches(
    dataset,
    batch_size,
    shuffle=True,
    drop_last=True,
    device='cpu'
    ):
    """
    This generator wraps the DataLoader class to build tensors out of the
    raw data and send them to the desired device
    """
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last)

    for data_dict in data_loader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict
