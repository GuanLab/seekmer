import bz2
import contextlib
import gzip
import io
import lzma
import pathlib
import subprocess

import logbook

from ._common import (INVALID_INDEX, KMerIndex)

__all__ = ('INVALID_INDEX', 'BUFFER_SIZE', 'decompress_and_open', 'read_fasta',
           'iterate_by_group', 'KMerIndex')

BUFFER_SIZE = 65536

_LOG = logbook.Logger('seekmer.common')

_FILE_MODE = 'rb'


@contextlib.contextmanager
def decompress_and_open(path):
    """Open the specified file in read-only binary mode.

    If the file is compressed, decompress the file according to the suffix.

    Parameters
    ----------
    path : pathlib.Path or str
        A file path

    Returns
    -------
    A file object pointing to the f
    """
    path = pathlib.Path(path)
    if path.suffix == '.gz':
        try:
            with subprocess.Popen(
                    ['zcat', str(path)], stdout=subprocess.PIPE,
            ) as process:
                yield process.stdout
        except OSError:
            _LOG.warn('Unable to call zcat, falling back to gzip module.')
            with gzip.open(str(path), _FILE_MODE) as raw, \
                    io.BufferedReader(raw) as f:
                yield f
    elif path.suffix == '.bz2':
        try:
            with subprocess.Popen(
                    ['bzcat', str(path)], stdout=subprocess.PIPE,
            ) as process:
                yield process.stdout
        except OSError:
            _LOG.warn('Unable to call bzcat, falling back to bz2 module.')
            with bz2.open(str(path), _FILE_MODE) as raw, \
                    io.BufferedReader(raw) as f:
                yield f
    elif path.suffix == '.xz' or path.suffix == '.lzma':
        try:
            with subprocess.Popen(
                    ['xzcat', str(path)], stdout=subprocess.PIPE,
            ) as process:
                yield process.stdout
        except OSError:
            _LOG.warn('Unable to call xzcat, falling back to lzma module.')
            with lzma.open(str(path), _FILE_MODE) as raw, \
                    io.BufferedReader(raw) as f:
                yield f
    else:
        with path.open(_FILE_MODE) as f:
            yield f


def read_fasta(path):
    """Iterate over entries in a FASTA file.

    Parameters
    ----------
    path : pathlib.Path
        A FASTA file

    Yields
    ------
    name : bytes
        The entry name
    sequence : bytes
        The sequence
    """
    name = None
    sequence = []
    with decompress_and_open(path) as file:
        for line in file:
            if line[0] != ord(b'>'):
                sequence.append(line.strip())
                continue
            if name is not None:
                yield name, b''.join(sequence)
            name = line[1:].strip()
            sequence.clear()
        if name is not None:
            yield name, b''.join(sequence)


def iterate_by_group(iterator, group_size):
    """Go through an iterator by groups.

    :param iterator:
    :param group_size:
    :return:
    """
    return zip(*(iter(iterator) * group_size))