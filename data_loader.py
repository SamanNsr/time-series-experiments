import numpy as np


class TimeSeriesBatchGenerator:
    def __init__(self, data, window_size, horizon=1, batch_size=32,
                 shuffle=True, drop_last=False, seed=None):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.data = data
        self.T, self.n_features = data.shape
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rng = np.random.RandomState(seed)

        self.n_samples = self.T - self.window_size - self.horizon + 1
        if self.n_samples <= 0:
            raise ValueError(
                "Not enough time steps for given window_size + horizon.")

        self.indices = np.arange(self.n_samples)
        self._reset_epoch_state()

    def _reset_epoch_state(self):
        self.pos = 0
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __iter__(self):
        self._reset_epoch_state()
        return self

    def __next__(self):
        if self.pos >= self.n_samples:
            raise StopIteration

        # select indices for this batch
        end = self.pos + self.batch_size
        batch_idx = self.indices[self.pos: end]
        # if last partial batch and drop_last True -> stop
        if len(batch_idx) < self.batch_size and self.drop_last:
            raise StopIteration

        B = len(batch_idx)
        X = np.empty((B, self.window_size, self.n_features),
                     dtype=self.data.dtype)
        Y = np.empty((B, self.horizon, self.n_features), dtype=self.data.dtype)

        for i, start in enumerate(batch_idx):
            X[i] = self.data[start: start + self.window_size]

            tgt_start = start + self.window_size
            tgt_end = tgt_start + self.horizon
            Y[i] = self.data[tgt_start: tgt_end]

        self.pos = end
        return X, Y

    def __len__(self):
        # number of batches per epoch (respecting drop_last)
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size
