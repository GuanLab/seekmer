__all__ = ('add_subcommand_parser', 'run', 'build', 'load_exome',
           'read_transcripts', 'extract_sequences',
           'extract_transcripts_from_genome', 'ContigAssembler')

import gc
import pathlib

import logbook
import numpy
import pandas

from . import common
from ._index_builder import (extract_sequences, ContigAssembler)

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
        transcript_ids, sequences = read_transcripts(fasta_path)
    else:
        transcript_ids, sequences = extract_transcripts_from_genome(
            fasta_path, exome,
        )
    index = build(transcript_ids, sequences, exome)
    index.save(index_path)


def build(transcript_ids, sequences, exome):
    """Build a Seekmer index for the given transcriptome.

    Parameters
    ----------
    transcript_ids : list[bytes]
        Transcriptome
    sequences : list[bytes]
        Transcriptome
    exome : numpy.recarray
        Transcriptome
    """
    if len(transcript_ids) == 0:
        raise ValueError('no transcripts found')
    transcriptome, exome = _compile_omics(transcript_ids, sequences, exome)
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
    table = table[[
        'transcript_id', 'gene_id', 'exon_number', 'chromosome', 'start',
        'end', 'strand',
    ]]
    table = table.to_records(index=False).astype([
        ('transcript_id', 'S{}'.format(transcript_id_length)),
        ('gene_id', 'S{}'.format(gene_id_length)),
        ('exon_number', 'i4'),
        ('chromosome', 'S{}'.format(chromosome_length)),
        ('start', 'i4'),
        ('end', 'i4'),
        ('strand', '?'),
    ])
    _LOG.info('Collected {} exon entries', len(table))
    return table


def read_transcripts(fasta):
    gc.collect()
    _LOG.info('Collecting transcripts')
    transcript_ids = []
    sequences = []
    for i, (id_, seq) in enumerate(common.read_fasta(fasta)):
        if i % 10000 == 0:
            _LOG.debug('Collected {} transcripts', i)
        transcript_ids.append(id_.split()[0].split(b'.')[0])
        sequences.append(seq)
    _LOG.info('Collected {} transcripts', len(transcript_ids))
    return transcript_ids, sequences


def extract_transcripts_from_genome(fasta, exome):
    _LOG.info('Collecting transcripts')
    transcript_ids = []
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
        transcript_ids.extend(new_transcripts)
        sequences.extend(new_sequences)
    _LOG.info('Collected {} transcripts', len(transcript_ids))
    return transcript_ids, sequences


def _compile_omics(transcript_ids, sequences, exome):
    transcript_id_length = max(len(id_) for id_ in transcript_ids)
    transcript_ids = numpy.asarray(transcript_ids,
                                   dtype='S{}'.format(transcript_id_length))
    exome = exome[numpy.in1d(exome['transcript_id'], transcript_ids)]
    exome.sort()
    transcript_index = numpy.searchsorted(exome['transcript_id'],
                                          transcript_ids)
    mask = (numpy.take(exome, transcript_index, mode="clip")['transcript_id']
            != transcript_ids)
    id_dtype = [(name, exome.dtype.fields[name][0])
                for name in ['transcript_id', 'gene_id']]
    transcriptome = numpy.zeros(len(transcript_ids),
                                dtype=id_dtype + [('length', 'f8')])
    transcriptome['transcript_id'] = transcript_ids
    gene_id = numpy.take(exome, transcript_index, mode="clip")['gene_id']
    transcriptome['gene_id'] = numpy.where(mask, b'', gene_id)
    transcriptome['length'] = [len(seq) for seq in sequences]
    return transcriptome, exome
