import torch
from torch import nn


class SurnameClassifier(torch.nn.Module):

    def __init__(self, char_embedding_dim, char_vocab_size, rnn_hidden_size, nb_categories, padding_idx):
        """
        char_embedding_dim: dimension of the space where characters will be embedded
        char_vocab_size: number of characters in vocabulary
        rnn_hidden_size: size of the hidden state vector in the rnn
        nb_categories: number of categories (labels)
        padding_idx: padding index in surname vectors
        """
        super(SurnameClassifier, self).__init__()

        self.embedding = nn.Embedding(
            embedding_dim=char_embedding_dim,
            num_embeddings=char_vocab_size,
            padding_idx=padding_idx
            )

        self.rnn = nn.GRU(
            input_size=char_embedding_dim,
            hidden_size=rnn_hidden_size,
            batch_first=True
        )

        self.fc_1 = nn.Linear(
            in_features=rnn_hidden_size,
            out_features=nb_categories
        )

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in):
        """
        x_in (batch_size, seq_len)
        example:
        [
            [<start_token>, 4, 5, 6, <end_token>, <padding_token>],
            [<start_token>, 3, 2, 1, <end_token>, <padding_token>]
        ]
        """
        # the embedding layer embeds each character into a space with dim = char_embedding_dim
        embedding = self.embedding(x_in)  # -> (batch_size, seq_len, char_embedding_size)

        # the RNN layer will result in an output vector and a hidden state
        # rnn_output -> (bacth_size, seq_len, rnn_hidden_size)
        # rnn_state -> (1, bacth_size, rnn_hidden_size)
        rnn_output, rnn_state = self.rnn(embedding)
        rnn_state = rnn_state.permute(1, 0, 2).flatten(start_dim=1) # rnn_state -> (bacth_size, rnn_hidden_size)
        categories = self.relu(self.fc_1(rnn_state)) # (batch_size x nb_categories)
        categories = self.softmax(categories) #(batch_size x nb_categories) as probs sum to one
        return categories
