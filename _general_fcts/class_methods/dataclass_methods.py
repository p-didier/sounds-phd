from copy import copy
import pickle, gzip
from pathlib import Path
from dataclasses import fields
from pathlib import PurePath
import json, os
import dataclass_wizard as dcw


def save(self, foldername: str, exportType='json'):
    """
    Saves program settings so they can be loaded again later
    
    Parameters
    ----------
    self : dataclass
        Dataclass to be exported.
    foldername : str
        Folder where to export the dataclass.
    exportType : str
        Type of export. "json": exporting to JSON file. "pkl": exporting to PKL.GZ archive.
    """

    shortPath = shorten_path(foldername, 3)
    if not Path(foldername).is_dir():
        Path(foldername).mkdir(parents=True)
        print(f'Created output directory ".../{shortPath}".')

    fullPath = f'{foldername}/{type(self).__name__}'
    if exportType == 'pkl':
        fullPath += '.pkl.gz'
        pickle.dump(self, gzip.open(fullPath, 'wb'))
    elif exportType == 'json':
        fullPath += '.json'
        save_to_json(self, fullPath)

    print(f'<{type(self).__name__}> object data exported to directory\n".../{shortPath}".')


def load(self, foldername: str, silent=False, dataType='json'):
    """
    Loads program settings object from file

    Parameters
    ----------
    self : dataclass
        Dataclass to be exported.
    foldername : str
        Folder where to export the dataclass.
    silent : bool
        If True, no printouts.
    dataType : str
        Type of file to import. "json": JSON file. "pkl": PKL.GZ archive.
    """
    shortPath = shorten_path(foldername, 3)
    if not Path(foldername).is_dir():
        raise ValueError(f'The folder "{foldername}" cannot be found.')

    if dataType == 'pkl':
        baseExtension = '.pkl.gz'
        altExtension = '.json'
    elif dataType == 'json':
        baseExtension = '.json'
        altExtension = '.pkl.gz'
    
    pathToFile = f'{foldername}/{type(self).__name__}{baseExtension}'
    if not Path(pathToFile).is_file():
        pathToAlternativeFile = f'{foldername}/{type(self).__name__}{altExtension}'
        if Path(pathToAlternativeFile).is_file():
            print(f'The file\n"{pathToFile}"\ndoes not exist. Loading\n"{pathToAlternativeFile}"\ninstead.')
            pathToFile = copy(pathToAlternativeFile)
            baseExtension = copy(altExtension)
        else:
            raise ValueError(f'Import issue, file\n"{pathToFile}"\nnot found (with either possible extensions).')

    if baseExtension == '.json':
        p = load_from_json(pathToFile, self)
    elif baseExtension == '.pkl.gz':
        p = pickle.load(gzip.open(pathToFile, 'r'))
    else:
        raise ValueError(f'Incorrect base extension: "{baseExtension}".')
    
    if not silent:
        print(f'<{type(self).__name__}> object data loaded from directory\n".../{shortPath}".')

    return p


def save_as_txt(self, filename):
    """Saves dataclass to TXT file"""

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
    f.close()


def save_to_json(mycls, filename):
    """Saves dataclass to JSON file"""

    # Check extension
    filename_alone, file_extension = os.path.splitext(filename)
    if file_extension != '.json':
        print(f'The filename ("{filename}") should be that of a JSON file. Modifying...')
        filename = filename_alone + '.json'
        print(f'Filename modified to "{filename}".')

    jsondict = dcw.asdict(mycls)  # https://stackoverflow.com/a/69059343

    with open(filename, 'w') as file_json:
        json.dump(jsondict, file_json, indent=4)
    file_json.close()


def load_from_json(path_to_json_file, mycls):
    """Loads dataclass from JSON file"""

    # Check extension
    filename_alone, file_extension = os.path.splitext(path_to_json_file)
    if file_extension != '.json':
        print(f'The filename ("{path_to_json_file}") should be that of a JSON file. Modifying...')
        path_to_json_file = filename_alone + '.json'
        print(f'Filename modified to "{path_to_json_file}".')

    with open(path_to_json_file) as fp:
        d = json.load(fp)
    mycls_out = dcw.fromdict(type(mycls), d)
    return mycls_out


def shorten_path(file_path, length=3):
    """Splits `file_path` into separate parts, select the last 
    `length` elements and join them again
    -- from: https://stackoverflow.com/a/49758154
    """
    return Path(*Path(file_path).parts[-length:])