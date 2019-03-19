import pathlib

import logbook
import numpy
import pandas
import scipy.stats
import sklearn.cluster

from . import common
from . import infer
from . import mapper

__all__ = ('add_subcommand_parser', 'run')

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
    parser.add_argument('-p', '--power', type=int, dest='power',
                        metavar='P', default=None,
                        help='specify the power of the weight matrix')
    parser.add_argument('-m', '--save-readmap', action='store_true',
                        dest='save_readmap', help='output an readmap file')
    parser.add_argument('-s', '--single-ended', action='store_true',
                        dest='single_ended',
                        help='specify whether the reads are single-ended')


def run(index_path, output_path, fastq_paths, job_count, single_ended, debug,
        power, **__):
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
    power : int
        The power to raise the weight matrix.
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
    _LOG.info('Mapped all reads.')
    _merge_fragment_lengths(map_results)
    map_results = [r.summarize() for r in map_results]
    _LOG.info('First round quantification...')
    base_results = numpy.asarray([infer.quantify(r) for r in map_results])
    abundances = {}
    if power is not None:
        _LOG.info('Weighting cells.')
        weight = _calculate_cell_weights(index, base_results) ** power
        _blend_mapping_results(map_results, weight)
        _LOG.info('Second round quantification...')
        if single_ended:
            for path, result in zip(fastq_paths, map_results):
                abundances[str(path)] = infer.quantify(result)
        else:
            for path, result in zip(fastq_paths[::2], map_results):
                abundances[str(path)] = infer.quantify(result)
    else:
        if single_ended:
            for path, result in zip(fastq_paths, base_results):
                abundances[str(path)] = result
        else:
            for path, result in zip(fastq_paths[::2], base_results):
                abundances[str(path)] = result
    id_ = numpy.char.decode(index.transcripts['transcript_id'])
    abundances = pandas.DataFrame(abundances, index=id_)
    _LOG.info('Writing results to {}...', output_path)
    abundances.to_csv(output_path / 'tpm.csv')


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
    for result in map_results:
        fragment_length_counts += result.fragment_length_counts
    for result in map_results:
        result.fragment_length_counts = fragment_length_counts


def _calculate_uniquely_mapped_counts(index, map_results):
    """Count uniquely mapped reads for all cells.

    Parameters
    ----------
    index : seekmer.KMerIndex
        The Seekmer index.
    map_results : list[seekmer.SummarizedResult]
        Mapping results of all single cells.

    Returns
    -------
    numpy.ndarray
        A cell-by-gene count matrix.
    """
    genes, transcript_gene_map = numpy.unique(index.transcripts['gene_id'],
                                              return_inverse=True)
    gene_read_counts = numpy.zeros((len(map_results), len(genes)), dtype='i8')
    for i, map_result in enumerate(map_results):
        classes = numpy.copy(map_result.class_map)
        read_count = map_result.class_count
        classes[1, :] = transcript_gene_map[classes[1, :]]
        classes = numpy.unique(classes, axis=0)
        __, class_index, class_count = numpy.unique(
            classes[0, :], return_inverse=True, return_counts=True,
        )
        classes = classes[:, class_count[class_index] == 1]
        if classes.size == 0:
            continue
        gene_read_counts[i] = numpy.bincount(
            classes[1, :], weights=read_count[class_count == 1],
            minlength=len(genes),
        )
    gene_mask = genes != b''
    return gene_read_counts[:, gene_mask]


def _calculate_cell_weights(index, base_matrix):
    """Estimate the cell weight matrix.

    Arguments
    ---------
    index : seekmer.KMerIndex
        The Seekmer index.
    base_matrix : numpy.ndarray
        A cell-by-gene count matrix.

    Returns
    -------
    numpy.ndarray
        A cell-by-cell weight matrix.
    """
    genes, transcript_gene_map = numpy.unique(index.transcripts['gene_id'],
                                              return_inverse=True)
    gene_matrix = numpy.zeros((len(base_matrix), len(genes)), dtype='i8')
    for i in range(len(genes)):
        gene_mask = transcript_gene_map == i
        gene_matrix[:, i] = base_matrix[:, gene_mask].sum(axis=1)
    gene_matrix = gene_matrix[:, genes != b'']
    weights = numpy.corrcoef(gene_matrix)
    flattened = weights[(weights == weights) & (weights != 1.0)]
    kmean = sklearn.cluster.KMeans(2)
    kmean.fit(flattened[:, None])
    weights[weights != weights] = 0.0
    filter_ = (kmean.predict(weights.flatten()[:, None])
               == kmean.cluster_centers_.argmax())
    weights = numpy.where(filter_.reshape(weights.shape), weights, 0.0)
    return weights


def _blend_mapping_results(map_results, weight):
    """Blend mapping results.

    Arguments
    ---------
    map_results : list[seekmer.SummarizedResult]
        Mapping results of all single cells.
    weight : numpy.ndarray
        A cell-by-cell weight matrix.
    """
    start = 0
    class_maps = []
    class_counts = []
    for result in map_results:
        result.class_map[0, :] += start
        start = result.class_map[0, :].max() + 1
        class_maps.append(result.class_map)
        class_counts.append(result.class_count)
    class_maps = numpy.concatenate(class_maps, axis=1)
    for i, result in enumerate(map_results):
        result.class_map = class_maps
        total_count = result.class_count.sum()
        result.class_count = numpy.concatenate(
            [c * w * total_count / c.sum()
             for c, w in zip(class_counts, weight[i, :])]
        )
