import bz2
import re

import numpy

from .. import common
from .. import index_builder


class TestIndexBuilder:

    def test_load_exome(self, shared_datadir):
        gtf_path = shared_datadir / 'human.ens90.21.gtf.gz'
        exome = index_builder.load_exome(gtf_path)
        assert isinstance(exome, numpy.recarray)
        assert len(exome) == 13977
        assert (exome['chromosome'] == b'21').all()

    def test_read_transcripts(self, shared_datadir):
        fasta_path = shared_datadir / 'human.cdna.21.fa.bz2'
        ids, sequences = index_builder.read_transcripts(fasta_path)
        fasta_file = bz2.open(str(shared_datadir / 'human.cdna.21.fa.bz2'))
        entry_count = sum([line[0] == ord(b'>') for line in fasta_file])
        id_set = set(ids)
        assert len(id_set) == len(ids)
        assert len(id_set) == entry_count
        assert len(id_set) == len(sequences)
        sequence_pattern = re.compile(br'^[ATCGatcgN]+$')
        for sequence in sequences:
            assert sequence_pattern.match(sequence) is not None

    def test_extract_transcripts_from_genome(self, shared_datadir):
        gtf_path = shared_datadir / 'human.ens90.21.gtf.gz'
        fasta_path = shared_datadir / 'human.dna.21.fa.gz'
        exome = index_builder.load_exome(gtf_path)
        ids, sequences = index_builder.extract_transcripts_from_genome(
            fasta_path, exome,
        )
        known_ids = numpy.unique(exome['transcript_id'])
        assert len(known_ids) == len(ids)
        assert len(known_ids) == len(sequences)
        sequence_pattern = re.compile(br'^[ATCGatcgN]+$')
        for sequence in sequences:
            assert sequence_pattern.match(sequence) is not None

    def test_build(self, shared_datadir):
        gtf_path = shared_datadir / 'human.ens90.21.gtf.gz'
        fasta_path = shared_datadir / 'human.cdna.21.fa.bz2'
        exome = index_builder.load_exome(gtf_path)
        ids, sequences = index_builder.read_transcripts(fasta_path)
        index = index_builder.build(ids, sequences, exome)
        assert isinstance(index, common.KMerIndex)
        kmer_entries = index.kmers['entry']
        kmer_entries = kmer_entries[kmer_entries != common.INVALID_INDEX]
        kmer_entries = numpy.where(kmer_entries < 0, ~kmer_entries,
                                   kmer_entries)
        assert kmer_entries.min() == 0
        contigs = index.contigs
        assert kmer_entries.max() == len(contigs) - 1
        assert contigs['offset'].min() == 0
        sequences = index.sequences
        max_contig_offset = (contigs['offset'] + contigs['length']).max()
        assert max_contig_offset == index.sequences.size
        assert b''.join(numpy.unique(sequences)) == b'ACGT'
        assert contigs['target_offset'].min() == 0
        max_contig_target_offset = (contigs['target_offset']
                                    + contigs['target_count']).max()
        targets = index.targets
        assert max_contig_target_offset == targets.size
        target_entries = targets['entry']
        target_entries = numpy.where(target_entries < 0, ~target_entries,
                                     target_entries)
        assert target_entries.min() == 0
        transcripts = index.transcripts
        assert target_entries.max() == transcripts.size - 1

    def test_build_with_extra_fasta(self, shared_datadir):
        gtf_path = shared_datadir / 'human.ens90.21.gtf.gz'
        fasta_path = shared_datadir / 'human.cdna.21.with_extra.fa.gz'
        exome = index_builder.load_exome(gtf_path)
        ids, sequences = index_builder.read_transcripts(fasta_path)
        index = index_builder.build(ids, sequences, exome)
        assert isinstance(index, common.KMerIndex)
        target_entries = index.targets['entry']
        target_entries = numpy.where(target_entries < 0, ~target_entries,
                                     target_entries)
        assert target_entries.min() == 0
        assert target_entries.max() == 2
        transcripts = index.transcripts
        assert transcripts.shape[0] == 3
