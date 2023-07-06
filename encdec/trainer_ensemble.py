import torch
import torch.nn.functional as F

import numpy as np

from trainer import Trainer
from tqdm import tqdm

class EnsembleTrainer(Trainer):
    def do_eval_rank_classification(self, model, data):
        all_losses = [] # loss based on *each* in-context example
        for dataloader in tqdm(data.dataloaders, desc="Ensemble"):
            losses = []
            for batch in tqdm(dataloader, desc="Eval (Rank)"):
                with torch.no_grad():
                    # self.print_tensor(batch[0], batch[2])
                    # breakpoint()
                    loss = self.run_model(model, batch, is_training=False)
                    losses += loss.cpu().detach().numpy().tolist()
            losses = np.array(losses)
            all_losses.append(losses)

        # stack together and take average
        losses = np.mean(np.stack(all_losses, axis=0), axis=0)

        predictions = []
        for idx, dp in enumerate(data.metadata):
            curr_instance_losses = [losses[indices] for indices in dp["indices"]]
            prediction_idx = sorted(enumerate(curr_instance_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())

        return predictions