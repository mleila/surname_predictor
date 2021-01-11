from surnames.constants import (
    START_TOKEN,
    END_TOKEN,
    MASK_TOKEN,
    UNK_TOKEN
    )


class OriginVocabulary:

    def __init__(self):
        self._token_to_index = {}
        self._tokens = []

    def add_token(self, token):
        if token in self._token_to_index:
            return
        self._token_to_index[token] = len(self._tokens)
        self._tokens.append(token)

    def lookup_token(self, origin):
        if origin not in self._token_to_index:
            raise KeyError(f'Origin {origin} not in vocabulary')
        return self._token_to_index[origin]

    def lookup_index(self, index):
        if index >= len(self._tokens):
            raise IndexError('index is beyond the number of tokens in this vocab')
        return self._tokens[index]

    def __len__(self):
        return len(self._tokens)


class SurnamesVocabulary:

    def __init__(
        self,
        start_token: str=START_TOKEN,
        end_token: str=END_TOKEN,
        mask_token: str=MASK_TOKEN,
        unknown_token: str=UNK_TOKEN
        ):
        self._char_to_index = {}
        self._characters = []

        # register special tokens
        self.start_token = start_token
        self.end_token = end_token
        self.mask_token = mask_token
        self.unknown_token = unknown_token

        # add special tokens
        self.add_token(start_token)
        self.add_token(end_token)
        self.add_token(mask_token)
        self.add_token(unknown_token)

    def add_token(self, token):
        if token in self._char_to_index:
            return
        self._char_to_index[token] = len(self._characters)
        self._characters.append(token)

    def lookup_token(self, token):
        if token not in  self._char_to_index:
            return self.lookup_token(self.unknown_token)
        return self._char_to_index[token]

    @property
    def start_index(self):
        return self.lookup_token(self.start_token)

    @property
    def end_index(self):
        return self.lookup_token(self.end_token)

    @property
    def unk_index(self):
        return self.lookup_token(self.unknown_token)

    @property
    def mask_index(self):
        return self.lookup_token(self.mask_token)

    def lookup_index(self, index):
        if index >= len(self._characters):
            raise IndexError('index is beyond the number of characters in this vocab')
        return self._characters[index]

    def __len__(self):
        return len(self._characters)
