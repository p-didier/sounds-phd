import pickle, gzip
from pathlib import Path
from dataclasses import fields
from pathlib import PurePath
import dataclasses, json, os

class EnhancedJSONEncoder(json.JSONEncoder):  # https://stackoverflow.com/a/51286749
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)
class EnhancedJSONDecoder(json.JSONDecoder):  # https://stackoverflow.com/a/51286749
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)


def save(self, foldername: str):
    """Saves program settings so they can be loaded again later"""
    shortPath = shorten_path(foldername, 3)
    if not Path(foldername).is_dir():
        Path(foldername).mkdir(parents=True)
        print(f'Created output directory ".../{shortPath}".')
    pickle.dump(self, gzip.open(f'{foldername}/{type(self).__name__}.pkl.gz', 'wb'))
    print(f'<{type(self).__name__}> object data exported to directory\n".../{shortPath}".')


def load(self, foldername: str, silent=False):
    """Loads program settings object from file"""
    shortPath = shorten_path(foldername, 3)
    if not Path(foldername).is_dir():
        raise ValueError(f'The folder "{foldername}" cannot be found.')
    p = pickle.load(gzip.open(f'{foldername}/{type(self).__name__}.pkl.gz', 'r'))
    if not silent:
        print(f'<{type(self).__name__}> object data loaded from directory\n".../{shortPath}".')
    return p


def save_as_txt(self, filename):
    if filename[-4:] != '.txt':
        if filename[-1] != '/':
            filename += '/'
        filename += f'{PurePath(filename).name}.txt'
    f = open(filename, 'w')
    f.write(f'{type(self).__name__} class fields\n\n')
    flds = [(fld.name, getattr(self, fld.name)) for fld in fields(self)]
    for ii in range(len(flds)):
        string = f'Field "{flds[ii][0]}" = {flds[ii][1]}\n'
        f.write(string)


def save_to_json(mycls, filename):
    """Saves dataclass to JSON file"""

    # Check extension
    filename_alone, file_extension = os.path.splitext(filename)
    if file_extension != '.json':
        print(f'The filename ("{filename}") should be that of a JSON file. Modifying...')
        filename = filename_alone + '.json'
        print(f'Filename modified to "{filename}".')

    jsonstr = json.dumps(mycls, cls=EnhancedJSONEncoder)  # https://stackoverflow.com/a/51286749
    with open(filename, 'w') as file_json:
        file_json.write(jsonstr)
    file_json.close()


def load_from_json(path_to_json_file, mycls):
    """Loads dataclass from JSON file"""

    # Check extension
    filename_alone, file_extension = os.path.splitext(path_to_json_file)
    if file_extension != '.json':
        print(f'The filename ("{path_to_json_file}") should be that of a JSON file. Modifying...')
        path_to_json_file = filename_alone + '.json'
        print(f'Filename modified to "{path_to_json_file}".')

    fp = open(path_to_json_file)
    json_file_as_dict = json.load(fp, cls=EnhancedJSONDecoder)
    mycls_out = mycls(**json_file_as_dict)

    return mycls_out


def shorten_path(file_path, length=3):
    """Splits `file_path` into separate parts, select the last 
    `length` elements and join them again
    -- from: https://stackoverflow.com/a/49758154
    """
    return Path(*Path(file_path).parts[-length:])