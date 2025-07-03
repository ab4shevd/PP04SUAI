from tqdm import tqdm

class ProgressBar:
    def __init__(self, total):
        self.pbar = tqdm(
            total=total,
            desc="Склейка карт",
            unit="фрагм",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

    def update(self):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()