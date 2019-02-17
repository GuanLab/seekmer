import datetime
import pathlib
import warnings

import logbook
import numpy
import scipy.stats

import seekmer.infer
from . import common
from . import mapper

__all__ = ('quantify',)

_LOG = logbook.Logger(__name__)


def add_subcommand_parser(subparsers):
    """Add an infer command to the subparsers.

    Parameters
    ----------
    subparsers : argparse Subparsers
        A subparser group
    """
    parser = subparsers.add_parser(
        'impute', help='impute transcript abundance for single-cell data',
        epilog='To pass single-cell files to Seekmer, you need to demultiplex '
               'the read files first. List pair-ended read files in the '
               'argument list. Seekmer treats every two files as a single '
               'cell. If the reads are single-ended, pass "-s" to the '
               'program, and Seekmer treats every single file as a single '
               'cell.'
    )
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


def run(index_path, output_path, fastq_paths, job_count, single_ended, debug,
        **__):
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
    single_ended : bool
        Whether the FASTQ files are single-ended reads.
    debug : bool
        Whether to enable debugging mode.
    """
    start_time = datetime.datetime.utcnow()
    try:
        output_path.mkdir(parents=True)
    except FileExistsError:
        _LOG.warn('The output folder exists. Overriding...')
    _LOG.info('Inferring transcript abundance')
    index = common.KMerIndex.load(index_path)
    _LOG.info('Mapping all reads')
    if single_ended:
        read_feeder = seekmer.infer.feed_single_ended_reads(fastq_paths)
    else:
        read_feeder = seekmer.infer.feed_pair_ended_reads(fastq_paths)
    map_result = map_reads(read_feeder, index, job_count=job_count,
                           debug=debug)
    _LOG.info('Mapped all reads')
    mean_fragment_length = map_result.harmonic_mean_fragment_length
    _LOG.info('Estimated fragment length: {:.2f}', mean_fragment_length)
    summarized_results = map_result.summarize()
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


def quantify(index, results, x0=None, bootstrap=False):
    """Estimate the transcript abundance.

    Parameters
    ----------
    index : seekmer.KMerIndex
        The Seekmer index.
    results : seekmer.SummarizedResults
    x0 : numpy.ndarray[float]
        The initial guess of the abundance.
    bootstrap : bool
        Whether to bootstrap reads before quantification.

    Returns
    -------
    numpy.ndarray
        The expression level of the transcripts.
    """
    if not bootstrap:
        _LOG.info('Aligned {} reads ({:.2%})', results.aligned,
                  results.aligned / results.total)
    if results.class_map.size == 0:
        return numpy.zeros(index.transcripts.size).astype('f8')
    if bootstrap:
        n = results.class_count.sum()
        p = results.class_count.astype('f8') / n
        class_count = scipy.stats.multinomial(n=n, p=p).rvs().flatten()
    else:
        class_count = results.class_count
    # NOTE: The lengths are intentionally not adjusted. Current tests showed
    # the traditional correction using read fragment length distribution gives
    # higher errors in the final estimation.
    transcript_length = results.effective_lengths.astype('f8')
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
