import collections
import datetime
import json
import pathlib
import queue
import shlex
import sys
import threading

import logbook
import numpy
import pandas
import tables

from . import common
from . import quantifier
from ._mapper import (MAX_FRAGMENT_LENGTH, ReadMapper)

__all__ = ('MAX_FRAGMENT_LENGTH', 'add_subcommand_parser', 'MapResult',
           'SummarizedResult', 'run', 'map_reads')

_LOG = logbook.Logger(__name__)

BUFFER_SIZE = 65536

EPS = numpy.finfo('f4').eps


def add_subcommand_parser(subparsers):
    """Add an infer command to the subparsers.

    Parameters
    ----------
    subparsers : argparse Subparsers
        A subparser group
    """
    parser = subparsers.add_parser('infer', help='infer transcript abundance')
    parser.add_argument('index_path', type=pathlib.Path, metavar='index',
                        help='specify a Seekmer index file')
    parser.add_argument('output_path', type=pathlib.Path, metavar='output',
                        help='specify a output folder')
    parser.add_argument('fastq_paths', type=pathlib.Path, metavar='fastq',
                        nargs='+', help='specify a FASTQ read file')
    parser.add_argument('-j', '--jobs', type=int, dest='job_count',
                        metavar='N', default=1,
                        help='specify the maximum parallel job number')
    parser.add_argument('-m', '--save-readmap', action='store_true',
                        dest='save_readmap', help='output an readmap file')
    parser.add_argument('-s', '--single-ended', action='store_true',
                        dest='single_ended',
                        help='specify whether the reads are single-ended')
    parser.add_argument('-b', '--bootstrap', type=int, dest='bootstrap',
                        default=0,
                        help='specify the number of bootstrapped estimation')


SummarizedResult = collections.namedtuple(
    'SummarizedResult',
    [
        'aligned',
        'unaligned',
        'total',
        'class_map',
        'class_count',
        'fragment_length_frequencies',
        'effective_lengths'
    ]
)


class MapResult:
    """A mapping result collection with a lock."""

    def __init__(self, index, readmap=None):
        """Create a mapping result collection.

        Parameters
        ----------
        index : seekmer.KMerIndex
            The K-mer index.
        readmap : io.TextIOWrapper | NoneType
            The readmap output file.
        """
        self.lock = threading.Lock()
        self.counter = collections.Counter()
        self.index = index
        self.readmap = readmap
        self.fragment_length_counts = numpy.zeros(MAX_FRAGMENT_LENGTH,
                                                  dtype='i8')

    def update(self, read_names, iterable):
        """Add mapping results.

        Parameters
        ----------
        read_names : list[bytes]
            A list of read names, used in the read map file.
        iterable : list[tuple[int]]
            A list of potential mappable targets.
        """
        self.counter.update(iterable)
        if self.readmap is not None:
            for read_name, targets in zip(read_names, iterable):
                ids = self.index.transcripts[targets, ]['transcript_id']
                print(read_name.decode(), *[id_.decode() for id_ in ids],
                      sep='\t', file=self.readmap)

    def summarize(self, index):
        """Summarize the results.

        Parameters
        ----------
        index : seekmer.KMerIndex
            A Seekmer index.

        Returns
        -------
        SummarizedResults
            The summarized results.
        """
        class_count = []
        class_map = []
        unaligned = self.counter.pop((), 0)
        for i, (targets, count) in enumerate(self.counter.items()):
            for target in targets:
                class_map.append((i, target))
            class_count.append(count)
        class_map = numpy.asarray(class_map).T
        class_count = numpy.asarray(class_count, dtype='f8')
        aligned = class_count.sum()
        return SummarizedResult(
            aligned=int(aligned),
            unaligned=int(unaligned),
            total=int(aligned + unaligned),
            class_map=class_map,
            class_count=class_count,
            fragment_length_frequencies=self.fragment_length_counts,
            effective_lengths=index.transcripts['length'].astype('f4'),
        )

    def merge_fragment_lengths(self, fragment_length_counts):
        """Merge fragment length counting.

        Parameters
        ----------
        fragment_length_counts : numpy.ndarray[int]
            An array of numbers of reads with different estimated
            fragment lengths.
        """
        self.fragment_length_counts += fragment_length_counts

    @property
    def harmonic_mean_fragment_length(self):
        """Calcualte the harmonic mean of fragment lengths.

        Returns
        -------
        float
            The harmonic mean of fragment lengths.
        """
        assert self.fragment_length_counts[0] == 0
        numerator = self.fragment_length_counts.sum()
        if numerator == 0:
            return 0
        denominator = (self.fragment_length_counts[1:].astype('f8')
                       / numpy.arange(1, MAX_FRAGMENT_LENGTH)).sum()
        return numerator / denominator

    def clear(self):
        """Clear the counter."""
        self.counter.clear()


def run(index_path, output_path, fastq_paths, job_count, save_readmap,
        single_ended, bootstrap, debug, **__):
    """The entrypoint of the inference module.

    The debugging mode disables parallel mapping.

    Parameters
    ----------
    index_path : pathlib.Path
        The index file.
    output_path : pathlib.Path
        The output folder.
    fastq_paths : list[pathlib.Path]
        The FASTQ files.
    job_count : int
        The number of working threads.
    save_readmap : bool
        Whether to save the read map file.
    single_ended : bool
        Whether the FASTQ files are single-ended reads.
    bootstrap : int
        The number of bootstrapped estimation.
    debug : bool
        Whether to enable debugging mode.
    """
    start_time = datetime.datetime.utcnow()
    try:
        output_path.mkdir(parents=True)
    except FileExistsError:
        _LOG.warn('The output folder exists. Overriding...')
    if save_readmap:
        readmap = (output_path / 'readmap.txt').open('wt')
    else:
        readmap = None
    _LOG.info('Inferring transcript abundance')
    index = common.KMerIndex.load(index_path)
    _LOG.info('Mapping all reads')
    if single_ended:
        read_feeder = feed_single_ended_reads(fastq_paths)
    else:
        read_feeder = feed_pair_ended_reads(fastq_paths)
    map_result = map_reads(read_feeder, index, job_count=job_count,
                           readmap=readmap, debug=debug)
    _LOG.info('Mapped all reads')
    mean_fragment_length = map_result.harmonic_mean_fragment_length
    _LOG.info('Estimated fragment length: {:.2f}', mean_fragment_length)
    summarized_results = map_result.summarize(index)
    _LOG.info('Quantifying transcripts')
    main_result = quantifier.quantify(index, summarized_results)
    _LOG.info('Quantified transcripts')
    bootstrapped_results = [
        quantifier.quantify(index, summarized_results, x0=main_result,
                            bootstrap=True)
        for __ in range(bootstrap)
    ]
    output_results(output_path, index, start_time, summarized_results,
                   main_result, bootstrapped_results)
    _LOG.info('Wrote results to {}'.format(output_path))


def map_reads(read_feeder, index, job_count=1, readmap=None, debug=False):
    """Map reads.

    Parameters
    ----------
    read_feeder : iterator of reads
        The read feeder.
    index : seekmer.KMerIndex
        The K-mer index.
    job_count : int
        The number of concurrent job threads.
    readmap : io.TextIOWrapper | NoneType
        The readmap output file. Default is None.
    debug : bool
        Whether to enable debugging mode.

    Returns
    -------
    MapResult
        The mapping results.
    """
    map_result = MapResult(index, readmap)
    try:
        if debug:
            ReadMapper(index, map_result)(read_feeder)
        else:
            reads_queue = queue.Queue(job_count * 2)
            threads = []
            for __ in range(job_count):
                thread = threading.Thread(target=ReadMapper(index, map_result),
                                          args=(iter(reads_queue.get, None),))
                threads.append(thread)
                thread.start()
            for batch in read_feeder:
                reads_queue.put(batch)
            for __ in range(job_count):
                reads_queue.put(None)
            for thread in threads:
                thread.join()
            threads.clear()
    finally:
        if readmap is not None:
            readmap.close()
    return map_result


def feed_single_ended_reads(paths):
    """Yield single-ended reads.

    Parameters
    ----------
    paths : list[pathlib.Path]
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
        with common.decompress_and_open(path) as file:
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


def feed_pair_ended_reads(paths):
    """Yield pair-ended reads.

    Parameters
    ----------
    paths : list[pathlib.Path]
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
    grouper = [iter(paths)] * 2
    for path1, path2 in zip(*grouper):
        with common.decompress_and_open(path1) as file1, \
                common.decompress_and_open(path2) as file2:
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


def output_results(output_path, index, start_time, results, main_abundance,
                   bootstrapped_abundance):
    """Output the quantification results.

    Parameters
    ----------
    output_path : pathlib.Path
        The path to an output folder.
    index : seekmer.KMerIndex
        The Seekmer index used for quantification
    start_time : datetime.datetime
        The time when Seekmer mapping started.
    results : seekmer.SummarizedResults
        The summarized results.
    main_abundance : numpy.ndarray[float]
        The estimated abundance of the transcripts.
    bootstrapped_abundance : list[numpy.ndarray[float]]
        The bootstrapped results.
    """
    run_info = _generate_run_info(bootstrapped_abundance, index, results,
                                  start_time)
    json.dump(run_info, (output_path / 'run_info.json').open('w'))
    est_counts = _infer_est_counts(index, results, main_abundance)
    _output_abundance_table(output_path, index, est_counts, main_abundance)
    _output_hdf5(output_path, index, results, run_info, est_counts,
                 bootstrapped_abundance)


def _generate_run_info(bootstrapped_abundance, index, results, start_time):
    class_target_count = numpy.bincount(results.class_map[0],
                                        minlength=results.class_count.size)
    unique_count = results.class_count[class_target_count == 1].sum()
    return {
        'n_targets': len(index.transcripts),
        'n_bootstraps': len(bootstrapped_abundance),
        'n_processed': results.total,
        'n_pseudoaligned': results.aligned,
        'n_unique': int(unique_count),
        'p_pseudoaligned': results.aligned / results.total,
        'p_unique': unique_count / results.total,
        'kallisto_version': '0.44.0',
        'index_version': 9000,
        'start_time': start_time.isoformat(sep=' '),
        'call': ' '.join([shlex.quote(arg) for arg in sys.argv]),
    }


def _output_abundance_table(output_path, index, est_counts, main_abundance):
    main_table = pandas.DataFrame()
    main_table['target_id'] = [
        id_.decode() for id_ in index.transcripts['transcript_id']
    ]
    main_table['length'] = index.transcripts['length']
    main_table['eff_length'] = index.transcripts['length'].astype('f4')
    main_table['est_count'] = est_counts
    main_table['tpm'] = main_abundance
    main_table.to_csv(str(output_path / 'abundance.tsv'), sep='\t',
                      index=False, float_format='%g')


def _infer_est_counts(index, results, main_abundance):
    """Estimate read counts for all transcripts.

    Parameters
    ----------
    index : seekmer.KMerIndex
        The Seekmer index used for quantification.
    results : seekmer.SummarizedResults
        The summarized mapping results.
    main_abundance : numpy.ndarray[float]
        The estimated abundance of transcripts.

    Returns
    -------
    numpy.ndarray[float]
        The estimated read counts for each transcripts.
    """
    est_counts = main_abundance * index.transcripts['length']
    est_counts *= results.aligned / est_counts.sum()
    return est_counts


def _output_hdf5(output_path, index, results, run_info, est_counts,
                 bootstrapped_abundance):
    """Generate the HDF5 output file.

    Parameters
    ----------
    output_path : pathlib.Path
        The path to an output folder.
    index : seekmer.KMerIndex
        The Seekmer index used for quantification
    results : seekmer.SummarizedResults
        The summarized results.
    run_info : dict
        The run_info dictionary.
    est_counts : numpy.ndarray[float]
        The estimated read counts for each transcripts.
    bootstrapped_abundance : list[numpy.ndarray[float]]
        The bootstrapped results.
    """
    hdf5_filter = tables.Filters()
    with tables.open_file(str(output_path / 'abundance.h5'), mode='w',
                          filters=hdf5_filter) as file:
        _output_hdf5_run_info(file, index, results, run_info)
        file.create_carray('/', 'est_counts', obj=est_counts.astype('f8'))
        if bootstrapped_abundance:
            group = file.create_group('/', 'bootstrap')
            for i, bootstrap in enumerate(bootstrapped_abundance):
                file.create_carray(group, 'bs{}'.format(i), obj=bootstrap)


def _output_hdf5_run_info(file, index, results, run_info):
    """Output aux group to the HDF5 file.

    Parameters
    ----------
    file : tables.File
        The 'aux' group.
    index : seekmer.KMerIndex
        The Seekmer index used for quantification
    results : seekmer.SummarizedResults
        The summarized results.
    run_info : dict
        The run_info dictionary.
    """
    group = file.create_group('/', 'aux')
    file.create_carray(group, 'call',
                       obj=numpy.frombuffer(run_info['call'].encode(),
                                            dtype='S1'))
    file.create_carray(group, 'index_version',
                       obj=numpy.asarray([run_info['index_version']]))
    file.create_carray(group, 'start_time',
                       obj=numpy.frombuffer(run_info['start_time'].encode(),
                                            dtype='S1'))
    file.create_carray(group, 'num_bootstrap',
                       obj=numpy.asarray([run_info['n_bootstraps']]))
    file.create_carray(group, 'num_processed',
                       obj=numpy.asarray([run_info['n_processed']]))
    file.create_carray(group, 'kallisto_version',
                       obj=numpy.frombuffer(
                           run_info['kallisto_version'].encode(), dtype='S1',
                       ))
    file.create_carray(group, 'ids', obj=index.transcripts['transcript_id'])
    file.create_carray(group, 'lengths', obj=index.transcripts['length'])
    file.create_carray(group, 'fld',
                       obj=results.fragment_length_frequencies.astype('i4'))
    file.create_carray(group, 'eff_lengths',
                       obj=results.effective_lengths.astype('f8'))
    file.create_carray(group, 'bias_observed',
                       obj=numpy.ones(4096, dtype='i4'))
    file.create_carray(group, 'bias_normalized',
                       obj=numpy.ones(4096, dtype='f8'))
