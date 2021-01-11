import time

import torch

from surnames.constants import VALID, X_DATA, Y_TARGET, TRAIN, VALID
from surnames.utils import (
    get_latest_model_checkpoint,
    load_checkpoint,
    save_checkpoint
)


class Trainer:
    """
    This class is responsible for training models and all the associated bookkeeping
    """

    def __init__(self, data_loader, optimizer, model, model_dir, loss_func, device):
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.model = model
        self.model_dir = model_dir
        self.loss_func = loss_func
        self.device = device


    def get_training_state(self):
        return {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
            }


    def run(self, nb_epochs, dataset, batch_size, checkpoint=None, verbose=False):

        if checkpoint:
            latest_checkpoint = get_latest_model_checkpoint(self.model_dir)
            if latest_checkpoint:
                map_location = torch.device(self.device)
                checkpoint_state = torch.load(latest_checkpoint, map_location=map_location)
                load_checkpoint(checkpoint_state, self.model)

        for epoch in range(nb_epochs):

            # create list for epoch losses
            training_loss = []

            # save model training checkpoint
            if checkpoint:
                training_state = self.get_training_state()
                timestamp = int(time.time())
                filename = f"{self.model_dir}/{timestamp}.pth.tar"
                save_checkpoint(training_state, filename=filename)

            # set model mode to train
            self.model.train()
            dataset.set_split(TRAIN)

            for batch_gen in self.data_loader(dataset, batch_size=batch_size):
                x_in, y_true = batch_gen[X_DATA], batch_gen[Y_TARGET]
                self.optimizer.zero_grad()
                y_pred = self.model(x_in)
                loss = self.loss_func(y_pred, y_true)
                loss_batch = loss.item()
                training_loss.append(loss_batch)
                loss.backward()
                self.optimizer.step()

            avg_training_loss = sum(training_loss)/len(training_loss)

            # set model mode to eval
            self.model.eval()
            dataset.set_split(VALID)
            valid_losses = []

            for batch_gen in self.data_loader(dataset, batch_size=batch_size):
                x_in, y_true = batch_gen[X_DATA], batch_gen[Y_TARGET]
                y_pred = self.model(x_in)
                loss = self.loss_func(y_pred, y_true)
                loss_batch = loss.item()
                valid_losses.append(loss_batch)

            avg_valid_loss = sum(valid_losses)/len(valid_losses)
            if verbose:
                print(f'Completed Epoch {epoch}')
                print(f'Training loss {avg_training_loss:.2f}')
                print(f'Validation loss {avg_valid_loss:.2f}')
                print('====')
