import functools
import operator
import os
import pathlib

DATA_DIR = pathlib.Path(__file__).parent.parent / 'data'


def create_path(*parts):
    return functools.reduce(operator.truediv, map(pathlib.Path, parts))

def get_path(*parts, must_exist=True):
    path = DATA_DIR / create_path(*parts)
    if must_exist and not path.exists():
        raise ValueError(f'Path "{path}" does not exist!')
    return path

def get_file(*parts, must_exist=True):
    path = get_path(*parts, must_exist=must_exist)
    if must_exist and not path.is_file():
        raise ValueError(f'Path "{path}" if not a file!')
    if path.is_dir():
        raise ValueError(f'Path "{path}" points to a directory, not a file!')
    return path

def get_dir(*parts, must_exist=True):
    path = get_data_path(*parts, must_exist=must_exist)
    if must_exist and not path.is_dir():
        raise ValueError(f'Path "{path}" is not a directory!')
    if path.is_file():
        raise ValueError(f'Path "{path}" points to a file, not a directory!')
    return path

def get_glob(*parts):
    return DATA_DIR.glob(str(create_path(*parts)))

def get_files(*parts):
    return (path for path in get_glob(*parts) if path.is_file())

def get_dirs(*parts):
    return (path for path in get_glob(*parts) if path.is_dir())
