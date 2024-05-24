import torch


class Metrics(object):
    '''Class to compute metric averages'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss, self.acc = 0, 0
        self.avg_loss, self.avg_acc = 0, 0
        self.sum_loss, self.sum_acc = 0, 0
        self.count = 0

    def update(self, loss, acc, n=1):
        self.loss, self.acc = loss, acc
        self.sum_loss += loss * n
        self.sum_acc += acc * n
        self.count += n
        self.avg_loss, self.avg_acc = self.sum_loss / self.count, self.sum_acc / self.count


class Log(object):
    '''Class to record loss & accuracy'''

    def __init__(self):
        self.train_accuracy = []
        self.train_loss = []
        self.val_accuracy = []
        self.val_loss = []

    def append(self, train, val):
        self.train_loss.append(train.avg_loss)
        self.train_accuracy.append(train.avg_acc)
        self.val_loss.append(val.avg_loss)
        self.val_accuracy.append(val.avg_acc)


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 device='cuda',  # Whether to use the GPU
                 early_stopping_patience=-1,  # The patience for early stopping
                 scheduler=None):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._device = device
        self._early_stopping_patience = early_stopping_patience
        self.checkpoint = None
        self.scheduler = scheduler

        self._model = self._model.to(device)
        self._crit = self._crit.to(device)

    def save_checkpoint(self):
        '''Overwriting checkpoint whenever validation loss is improved'''
        torch.save(self._model.state_dict(), f'best.pth')

    def restore_best_checkpoint(self):
        '''Restore model weights of epoch with best validation loss'''
        self._model.load_state_dict(torch.load('./best.pth'))

    def train_step(self, input_ids, attention_mask, targets):

        # Reset gradients to zero
        self._optim.zero_grad()

        # Forward pass: compute predicted y
        y_hat = self._model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(y_hat, dim=1)

        # Compute loss
        loss = self._crit(y_hat, targets)

        # Calculate Metric (accuracy)
        acc = self.metric_calc(preds, targets)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update model parameters
        self._optim.step()

        # Return loss and predicted y
        return loss, y_hat, acc

    def val_test_step(self, input_ids, attention_mask, targets):

        # Forward pass: compute predicted y
        y_hat = self._model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute validation loss
        loss = self._crit(y_hat, targets)
        preds = torch.argmax(y_hat, dim=1)

        # Compute Metric (accuracy)
        acc = self.metric_calc(preds, targets)

        # Return validation loss and predicted y
        return loss, y_hat, acc

    def train_epoch(self):

        # Set training mode
        self._model.train()

        metrics = Metrics()

        # Iterate over batches in the training set
        for i, batch in enumerate(self._train_dl):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            targets = batch['targets']

            # Move tensors to GPU if available
            input_ids = input_ids.to(self._device, non_blocking=True).long()
            attention_mask = attention_mask.to(self._device, non_blocking=True).long()
            targets = targets.to(self._device, non_blocking=True).long()

            loss, preds, acc = self.train_step(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               targets=targets)

            metrics.update(loss.item(), acc, self._train_dl.batch_size)

        print(f"\tTraining Loss: {metrics.avg_loss}, Training Accuracy: {metrics.avg_acc}")

        return metrics

    def val_test(self):

        # Set evaluation mode
        self._model.eval()

        metrics = Metrics()

        # Disable gradient computation
        with torch.no_grad():
            # Iterate over batches in the training set
            for i, batch in enumerate(self._val_test_dl):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                targets = batch['targets']

                input_ids = input_ids.to(self._device, non_blocking=True).long()
                attention_mask = attention_mask.to(self._device, non_blocking=True).long()
                targets = targets.to(self._device, non_blocking=True).long()

                loss, preds, acc = self.val_test_step(input_ids=input_ids,
                                                      attention_mask=attention_mask,
                                                      targets=targets)

                metrics.update(loss.item(), acc, self._val_test_dl.batch_size)

        print(f"\t Validation Loss: {metrics.avg_loss}, Validation Accuracy: {metrics.avg_acc}")

        return metrics

    def metric_calc(self, preds, labels):
        # Accuracy
        correct = 0
        for pred, label in zip(preds, labels):
            if pred == label:
                correct += 1
        return correct / len(preds)

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        log = Log()  # Saving Loss and Accuracies
        epoch = 1
        no_improvements = 0
        best_loss = 10 ** 5
        epoch_best = None

        # Continue training until stopping condition is met
        while True:

            # Stop by epoch number
            if epoch > epochs:
                self.restore_best_checkpoint()
                print(f"The lowest validation loss was achieved in epoch {epoch_best}")
                break

            # Train the model for one epoch and record training and validation losses
            print(f'Epoch: {epoch}')
            train_metrics = self.train_epoch()
            val_metrics = self.val_test()
            log.append(train_metrics, val_metrics)

            # Adjust learning rate using scheduler if available
            if self.scheduler:
                self.scheduler.step(val_metrics.avg_loss)

            # Save the checkpoint if validation loss is lowest
            if val_metrics.avg_loss < best_loss:
                self.save_checkpoint()
                no_improvements = 0
                best_loss = val_metrics.avg_loss
                epoch_best = epoch
            else:
                no_improvements += 1

            # Check if early stopping condition met
            if no_improvements >= self._early_stopping_patience:
                self.restore_best_checkpoint()
                print(f"Early stopping. The lowest validation loss was achieved in epoch {epoch_best}")
                break
            epoch += 1

        return log