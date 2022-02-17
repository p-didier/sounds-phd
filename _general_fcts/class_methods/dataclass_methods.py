import pickle, gzip
from pathlib import Path


def save(self, foldername: str):
    """Saves program settings so they can be loaded again later"""
    if not Path(foldername).is_dir():
        Path(foldername).mkdir(parents=True)
        print(f'Created output directory "{foldername}".')
    pickle.dump(self, gzip.open(f'{foldername}/{type(self).__name__}.pkl.gz', 'wb'))
    print(f'<{type(self).__name__}> object data exported to directory\n"{foldername}".')


def load(self, foldername: str):
    """Loads program settings object from file"""
    if not Path(foldername).is_dir():
        raise ValueError(f'The folder "{foldername}" cannot be found.')
    p = pickle.load(gzip.open(f'{foldername}/{type(self).__name__}.pkl.gz', 'r'))
    print(f'<{type(self).__name__}> object data loaded from directory\n"{foldername}".')
    return p