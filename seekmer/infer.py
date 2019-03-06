import datetime
import json
import pathlib
import shlex
import sys
import warnings

import logbook
import numpy
import pandas
import scipy.stats
import tables

from . import mapper
from . import common

__all__ = [
    'run',
    'quantify',
    'em',
    'output_results',
]

_LOG = logbook.Logger(__name__)


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
        read_feeder = common.feed_single_ended_reads(*fastq_paths)
    else:
        read_feeder = common.feed_pair_ended_reads(*fastq_paths)
    map_result = mapper.map_reads(index, read_feeder, job_count=job_count,
                                  readmap=readmap, debug=debug)
    _LOG.info('Mapped all reads')
    mean_fragment_length = map_result.harmonic_mean_fragment_length
    _LOG.info('Estimated fragment length: {:.2f}', mean_fragment_length)
    summarized_results = map_result.summarize()
    _LOG.info('Quantifying transcripts')
    _LOG.info('Aligned {} reads ({:.2%})', summarized_results.aligned,
              summarized_results.aligned / summarized_results.total)
    main_result = quantify(summarized_results)
    _LOG.info('Quantified transcripts')
    bootstrapped_results = [
        quantify(summarized_results, x0=main_result, bootstrap=True)
        for __ in range(bootstrap)
    ]
    output_results(output_path, index, start_time, summarized_results,
                   main_result, bootstrapped_results)
    _LOG.info('Wrote results to {}'.format(output_path))


def quantify(results, x0=None, bootstrap=False):
    """Estimate the transcript abundance.

    Parameters
    ----------
    results : seekmer.SummarizedResults
        Summarized mapping results from the mapper.
    x0 : numpy.ndarray[float]
        The initial guess of the abundance.
    bootstrap : bool
        Whether to bootstrap reads before quantification.

    Returns
    -------
    numpy.ndarray
        The expression level of the transcripts.
    """
    transcript_length = results.effective_lengths.astype('f8')
    if results.class_map.size == 0:
        return numpy.zeros(results.effective_lengths.size).astype('f8')
    if bootstrap:
        n = results.class_count.sum()
        p = results.class_count.astype('f8') / n
        class_count = scipy.stats.multinomial(n=n, p=p).rvs().flatten()
    else:
        class_count = results.class_count
    if x0 is None:
        x = numpy.ones(transcript_length.size, dtype='f8') / transcript_length
    else:
        x = x0.copy()
    x /= x.sum()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', 'divide by zero encountered in true_divide',
        )
        warnings.filterwarnings(
            'ignore', 'invalid value encountered in true_divide',
        )
        x = em(x, transcript_length, results.class_map, class_count)
    x /= x.sum() / 1000000
    x[x < 0.001] = 0
    x /= x.sum() / 1000000
    return x


def em(x, l, class_map, class_count):
    """Expectation-maximization.

    Parameters
    ----------
    x: numpy.ndarray[float]
        The initial guess of the abundance.
    l: numpy.ndarray[float]
        The harmonic mean of the fragment lengths
    class_map : numpy.ndarray[int]
        An array of equivalent classes and their mappable targets.
    class_count : numpy.ndarray[int]
        The number of reads in each class.

    Returns
    -------
    numpy.ndarray[float]
        The expression level of the transcripts.
    """
    n = class_count.sum()
    old_x = x
    x = x[class_map[1]]
    class_inner = numpy.bincount(class_map[0], weights=x,
                                 minlength=class_count.size) / class_count
    x = numpy.bincount(class_map[1], weights=x / class_inner[class_map[0]],
                       minlength=l.size) / l / n
    x[x != x] = 0
    while (numpy.absolute(x - old_x) / x)[x > 1e-8].max() > 0.01:
        old_x = x
        x = x[class_map[1]]
        class_inner = numpy.bincount(class_map[0], weights=x,
                                     minlength=class_count.size) / class_count
        x = numpy.bincount(class_map[1], weights=x / class_inner[class_map[0]],
                           minlength=l.size) / l / n
        x[x != x] = 0
    return x


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
    _output_abundance_table(output_path, index, results, est_counts,
                            main_abundance)
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


def _output_abundance_table(output_path, index, results, est_counts,
                            main_abundance):
    main_table = pandas.DataFrame()
    main_table['target_id'] = [
        id_.decode() for id_ in index.transcripts['transcript_id']
    ]
    main_table['length'] = index.transcripts['length']
    main_table['eff_length'] = results.effective_lengths.astype('f4')
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
