__all__ = ('add_subcommand_parser', 'run', 'build', 'load_exome',
           'extract_transcripts', 'extract_sequences',
           'extract_transcripts_from_genome', 'ContigAssembler')


import gc
import pathlib

import logbook
import numpy
import pandas

from ._index_builder import (extract_sequences, ContigAssembler)
from . import common


_LOG = logbook.Logger(__name__)

_GTF_COLUMNS = [0, 2, 3, 4, 6, 8]

_GTF_DTYPES = {
    'chromosome': str,
    'feature': str,
    'start': int,
    'end': int,
    'strand': bool,
    'attributes': str,
}

_GTF_NAMES = [
    'chromosome',
    'feature',
    'start',
    'end',
    'strand',
    'attributes',
]

_GTF_ATTRIBUTES = [
    'transcript_id',
    'gene_id',
    'exon_number',
]


def add_subcommand_parser(subparsers):
    """Add an index command to the subparsers.

    Parameters
    ----------
    subparsers : argparse Subparsers
        A subparser group
    """
    parser = subparsers.add_parser('index', help='build a Seekmer index')
    parser.add_argument('-t', '--transcriptome',
                        action='store_true', dest='use_transcriptome',
                        help='use transcriptomic sequences instead of '
                             'genomic sequences')
    parser.add_argument('fasta_path', type=pathlib.Path, metavar='fasta',
                        help='specify a genome sequence file')
    parser.add_argument('gtf_path', type=pathlib.Path, metavar='gtf',
                        help='specify a transcript annotation file')
    parser.add_argument('index_path', type=pathlib.Path, metavar='index',
                        help='specify an output index file')


def run(fasta_path, gtf_path, index_path, use_transcriptome=False, **__):
    """Generate a Seekmer index file of the given transcriptome.

    Parameters
    ----------
    fasta_path : pathlib.Path
        The transcriptome sequence (such as ENSEMBL cDNA library FASTA
        file) if `use_transcriptome` is `True`, or the genome sequence
        if `use_transcriptome` is `False`.
    gtf_path : pathlib.Path
        The annotation GTF file.
    index_path : pathlib.Path
        The output index file.
    use_transcriptome : bool
        Whether the `fasta_path` contains the transcriptomic sequences
        or the genomic sequences.
    """
    _LOG.info('Building index')
    exome = load_exome(gtf_path)
    if use_transcriptome:
        transcriptome, sequences = extract_transcripts(fasta_path)
    else:
        transcriptome, sequences = extract_transcripts_from_genome(fasta_path,
                                                                   exome)
    index = build(transcriptome, sequences, exome)
    index.save(index_path)


def build(transcriptome, sequences, exome):
    """Build a Seekmer index for the given transcriptome.

    Parameters
    ----------
    transcriptome : numpy.recarray
        Transcriptome
    sequences : list of bytes
        Transcriptome
    exome : numpy.recarray
        Transcriptome
    """
    if len(transcriptome) == 0:
        raise ValueError('no transcripts found')
    transcriptome, sequences, exome = _compile_omics(transcriptome, sequences,
                                                     exome)
    if len(exome) == 0:
        raise RuntimeError('no matching exon records')
    assembler = ContigAssembler()
    kmer_table, contigs, sequences, targets = assembler.assemble(sequences)
    return common.KMerIndex(kmer_table, contigs, sequences, targets,
                            transcriptome, exome)


def load_exome(path):
    """Load exon records from the given GTF annotation file.

    Parameters
    ----------
    path : pathlib.Path
        The GTF file.

    Returns
    -------
    exome : numpy.recarray
        The exon table.
    """
    gc.collect()
    _LOG.info('Collecting exon entries from {}', path)
    table = pandas.read_table(str(path), comment='#', header=None,
                              usecols=_GTF_COLUMNS, names=_GTF_NAMES,
                              dtype=_GTF_DTYPES, true_values=['+'],
                              false_values=['-'])
    table.drop(table.index[table.feature != 'exon'], axis=0, inplace=True)
    table.start -= 1
    for attr in _GTF_ATTRIBUTES:
        table[attr] = table.attributes.str.replace(
            '.*{} "([^"]+)".*'.format(attr), r'\1',
        )
    # ERCC GTF does not have exon_number entries. They need to be fixed.
    table['exon_number'] = (pandas.to_numeric(table.exon_number, 'coerce')
                            .fillna(1).astype('i4'))
    table.drop(['feature', 'attributes'], axis=1, inplace=True)
    table.sort_values(
        ['chromosome', 'gene_id', 'transcript_id', 'exon_number'],
        inplace=True,
    )
    table['chromosome'] = table['chromosome'].str.encode('utf-8')
    chromosome_length = table['chromosome'].str.len().max()
    table['transcript_id'] = table['transcript_id'].str.encode('utf-8')
    transcript_id_length = table['transcript_id'].str.len().max()
    table['gene_id'] = table['gene_id'].str.encode('utf-8')
    gene_id_length = table['gene_id'].str.len().max()
    table = table.to_records(index=False).astype([
        ('chromosome', 'S{}'.format(chromosome_length)),
        ('start', 'i4'),
        ('end', 'i4'),
        ('strand', '?'),
        ('transcript_id', 'S{}'.format(transcript_id_length)),
        ('gene_id', 'S{}'.format(gene_id_length)),
        ('exon_number', 'i4'),
    ])
    _LOG.info('Collected {} exon entries', len(table))
    return table


def extract_transcripts(fasta):
    gc.collect()
    _LOG.info('Collecting transcripts')
    transcriptome = []
    sequences = []
    for i, (id_, seq) in enumerate(common.read_fasta(fasta)):
        if i % 10000 == 0:
            _LOG.debug('Collected {} transcripts', i)
        transcriptome.append(id_.split()[0].split(b'.')[0])
        sequences.append(seq)
    _LOG.info('Collected {} transcripts', len(transcriptome))
    return transcriptome, sequences


def extract_transcripts_from_genome(fasta, exome):
    _LOG.info('Collecting transcripts')
    transcriptome = []
    sequences = []
    success = False
    for i, (chromosome, sequence) in enumerate(common.read_fasta(fasta)):
        if i % 10 == 0 and not success and i != 0:
            _LOG.warning('Seekmer could not collect any exons in the first {} '
                         'chromosomes', i)
            _LOG.warning('If you are using transcriptomic sequences, please '
                         'add the "-t" option to the index command.')
        gc.collect()
        chromosome = chromosome.split()[0]
        targets = exome[exome['chromosome'] == chromosome]
        if len(targets) == 0:
            continue
        success = True
        new_transcripts, new_sequences = extract_sequences(targets, sequence)
        _LOG.debug('Collected {} transcripts from Chromosome {}',
                   len(new_sequences), chromosome)
        transcriptome.extend(new_transcripts)
        sequences.extend(new_sequences)
    _LOG.info('Collected {} transcripts', len(transcriptome))
    return transcriptome, sequences


def _compile_omics(transcriptome, sequences, exome):
    exome = exome[numpy.in1d(exome['transcript_id'], transcriptome)]
    common_transcripts, exon_index = numpy.unique(exome['transcript_id'],
                                                  return_index=True)
    print('filtered_exon:', len(exon_index))
    exon_index.sort()
    transcript_filter = numpy.in1d(transcriptome, common_transcripts)
    print('sequences:', len(sequences))
    print('filtersize:', len(transcript_filter))
    sequences = [
        seq for seq, check in zip(sequences, transcript_filter) if check
    ]
    id_dtype = [(name, exome.dtype.fields[name][0])
                for name in ['transcript_id', 'gene_id']]
    transcriptome = numpy.zeros(len(exon_index),
                                dtype=id_dtype + [('length', 'f8')])
    transcriptome['transcript_id'] = exome[exon_index]['transcript_id']
    transcriptome['gene_id'] = exome[exon_index]['gene_id']
    transcriptome['length'] = [len(seq) for seq in sequences]
    return transcriptome, sequences, exome
