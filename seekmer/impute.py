import pathlib

import logbook
import numpy
import scipy.stats

from . import common
from . import infer
from . import mapper

__all__ = ('add_subcommand_parser', '')

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
    try:
        output_path.mkdir(parents=True)
    except FileExistsError:
        _LOG.warn('The output folder exists. Overriding...')
    _LOG.info('Inferring transcript abundance')
    index = common.KMerIndex.load(index_path)
    _LOG.info('Mapping all reads')
    if single_ended:
        map_results = mapper.map_multiple_samples(
            index, [common.feed_single_ended_reads(path)
                    for path in fastq_paths],
            job_count=job_count, debug=debug,
        )
    else:
        map_results = mapper.map_multiple_samples(
            index, [common.feed_pair_ended_reads(*paths)
                    for paths in common.iterate_by_group(fastq_paths, 2)],
            job_count=job_count, debug=debug,
        )
    _LOG.info('Mapped all reads')
    _merge_fragment_lengths(map_results)
    map_results = [result.summarize() for result in map_results]
    weight = _calculate_weight_matrix(index, map_results)
    return
    map_results = _blend_mapping_results(map_results, weight)
    abundances = {}
    if single_ended:
        for path, result in zip(fastq_paths, map_results):
            abundances[str(path)] = infer.quantify(result)
    else:
        for path, result in zip(fastq_paths[::2], map_results):
            abundances[str(path)] = infer.quantify(result)


def _merge_fragment_lengths(map_results):
    """Combine all fragment length counts.

    Single cell data are often from a single sequencing batch, fragment
    length of which can be combined for better length adjustment.

    Parameters
    ----------
    map_results : list[seekmer.MapResult]
        Mapping results of all single cells.
    """
    fragment_length_counts = numpy.zeros(mapper.MAX_FRAGMENT_LENGTH,
                                         dtype='i8')
    for __, result in map_results:
        fragment_length_counts += result.fragment_length_counts
    for __, result in map_results:
        result.fragment_length_counts = fragment_length_counts


def _calculate_weight_matrix(index, map_results):
    """Estimate the read contribution between cells.

    Parameters
    ----------
    index : seekmer.KMerIndex
        The Seekmer index.
    map_results : list[seekmer.SummarizedResult]
        Mapping results of all single cells.

    Returns
    -------
    numpy.ndarray
        A cell-by-cell matrix that encodes weight contribution.
    """
    genes, transcript_gene_map = numpy.unique(index.transcripts['gene_id'],
                                              return_inverse=True)
    gene_read_counts = numpy.zeros((len(map_results), len(genes)), dtype='i8')
    for i, map_result in enumerate(map_results):
        classes = numpy.copy(map_result.class_map)
        read_count = map_result.class_count
        classes[:, 1] = transcript_gene_map[classes[:, 1]]
        classes = numpy.unique(classes, axis=0)
        class_count, class_index = numpy.unique(
            classes[:, 0], return_counts=True, return_inverse=True,
        )[1:]
        classes = classes[class_count[class_index] == 1]
        gene_read_counts[i] = numpy.bincount(
            classes[:, 1], weights=read_count[class_count == 1],
            minlength=len(genes),
        )
    gene_mask = genes != b''
    gene_read_counts = gene_read_counts[:, gene_mask]
    correlation = scipy.stats.spearmanr(gene_read_counts)[0]
    print(correlation)
    print(correlation[0])


def _blend_mapping_results(map_results, weight):
    pass
