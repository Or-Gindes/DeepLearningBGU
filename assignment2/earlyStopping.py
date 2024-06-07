import torch

class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, validation_loss, model):
        if self.best_score is None or self.best_score > validation_loss:
            self.save_checkpoint(validation_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == 20:
                self.early_stop = True

    def save_checkpoint(self,validation_loss: float, model):
        torch.save(model.state_dict(), f'model_val_loss_{validation_loss}.pt')