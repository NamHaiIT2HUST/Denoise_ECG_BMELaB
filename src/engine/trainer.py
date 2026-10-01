from pathlib import Path
import copy
import pandas as pd
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, device, run_dir):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _step(self, batch):
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)
        pred = self.model(x)
        loss = self.criterion(pred, y)
        return pred, loss

    def fit(self, train_loader, val_loader, epochs=50, patience=10):
        best = {'epoch': -1, 'val_loss': 1e18, 'state_dict': None}
        wait = 0
        train_logs = []
        val_logs = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f'Train {epoch}/{epochs}'):
                self.optimizer.zero_grad()
                _, loss = self._step(batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= max(1, len(train_loader))

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Val {epoch}/{epochs}'):
                    _, loss = self._step(batch)
                    val_loss += loss.item()
            val_loss /= max(1, len(val_loader))

            train_logs.append({'epoch': epoch, 'loss': train_loss})
            val_logs.append({'epoch': epoch, 'loss': val_loss})
            print(f'Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}')

            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict()}, self.run_dir / 'checkpoint_last.pt')
            if val_loss < best['val_loss']:
                best = {'epoch': epoch, 'val_loss': val_loss, 'state_dict': copy.deepcopy(self.model.state_dict())}
                torch.save({'epoch': epoch, 'model_state_dict': best['state_dict']}, self.run_dir / 'checkpoint_best.pt')
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print('Early stopping.')
                    break

        pd.DataFrame(train_logs).to_csv(self.run_dir / 'train_log.csv', index=False)
        pd.DataFrame(val_logs).to_csv(self.run_dir / 'val_log.csv', index=False)
        if best['state_dict'] is not None:
            self.model.load_state_dict(best['state_dict'])
        return best
