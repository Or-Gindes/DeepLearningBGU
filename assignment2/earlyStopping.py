import torch


class EarlyStopping:
    def __init__(self, patience=20):
        self.epoch = 0
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, validation_loss, model):
        if self.best_score is None or self.best_score >= validation_loss:
            self.best_score = validation_loss
            self.save_checkpoint(model)
            self.counter = 0
            self.epoch = epoch
        else:
            self.counter += 1
            if self.counter == 20:
                self.early_stop = True

    @staticmethod
    def save_checkpoint(model):
        torch.save(model.state_dict(), f'model_checkpnt.pt')
