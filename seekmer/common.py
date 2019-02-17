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

_LOG = logbook.Logger(__name__)

_FILE_MODE = 'rb'


@contextlib.contextmanager
def decompress_and_open(path):
    """Open the specified file in read-only binary mode.

    If the file is compressed, decompress the file according to the
    suffix.

    Parameters
    ----------
    path : pathlib.Path
        A file path

    Returns
    -------
    io.BufferedIOBase
        A file object
    """
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

    Parameters
    ----------
    iterator : collections.Iterable
        An iterable sequence
    group_size : int
        The size of grouped

    Returns
    -------
    collections.Iterable
        An iterable sequence that yields groups of elements
    """
    return zip(*([iter(iterator)] * group_size))


def feed_single_ended_reads(*paths):
    """Yield single-ended reads.

    Parameters
    ----------
    paths: list[pathlib.Path]
        The FASTQ files.

    Yields
    ------
    int
        The number of reads.
    list[bytes]
        The read names.
    list[bytes]
        The reads.
    """
    read_names = []
    reads = []
    for path in paths:
        with decompress_and_open(path) as file:
            for i, line in enumerate(file):
                if i & 3 == 0:
                    read_names.append(line.strip()[1:])
                elif i & 3 == 1:
                    reads.append(line.strip())
                    if len(read_names) >= BUFFER_SIZE:
                        yield len(read_names), read_names, reads
                        read_names = []
                        reads = []
    if reads:
        yield len(read_names), read_names, reads
    _LOG.debug('Finished reading sequence file(s)')


def feed_pair_ended_reads(*paths):
    """Yield pair-ended reads.

    Parameters
    ----------
    paths: list[pathlib.Path]
        The FASTQ files.

    Yields
    ------
    int
        The number of reads.
    list[bytes]
        The read names.
    list[bytes]
        The read pairs.
    """
    if len(paths) % 2 != 0:
        raise ValueError('cannot process odd numbers of pair-ended files')
    read_names = []
    reads = []
    for path1, path2 in iterate_by_group(paths, 2):
        with decompress_and_open(path1) as file1, \
                decompress_and_open(path2) as file2:
            for i, (line1, line2) in enumerate(zip(file1, file2)):
                if i & 3 == 0:
                    read_names.append(line1.strip()[1:])
                elif i & 3 == 1:
                    reads.append(line1.strip())
                    reads.append(line2.strip())
                    if len(read_names) >= BUFFER_SIZE:
                        yield len(read_names), read_names, reads
                        read_names = []
                        reads = []
    if reads:
        yield len(read_names), read_names, reads
    _LOG.debug('Finished reading sequence file(s)')
