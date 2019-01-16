import pytest

from .. import common
from .. import index_builder
from .. import mapper


_BASE_SET = set(b'ACTGNactg')


@pytest.fixture
def seekmer_index(shared_datadir):
    gtf_path = shared_datadir / 'human.ens90.21.gtf.gz'
    fasta_path = shared_datadir / 'human.cdna.21.fa.bz2'
    exome = index_builder.load_exome(gtf_path)
    ids, sequences = index_builder.read_transcripts(fasta_path)
    index = index_builder.build(ids, sequences, exome)
    return index


class TestReadFeeder:

    def test_feed_single_ended_reads(self, shared_datadir):
        path = shared_datadir / '20_1.fastq'
        idx = 0
        for count, names, reads in mapper.feed_single_ended_reads([path]):
            idx += 1
            assert count == 21
            assert len(names) == count
            assert len(reads) == count
            for read in reads:
                assert set(read) <= _BASE_SET
        assert idx == 1

    def test_feed_pair_ended_reads(self, shared_datadir):
        paths = [shared_datadir / '20_1.fastq', shared_datadir / '20_2.fastq']
        idx = 0
        for count, names, reads in mapper.feed_pair_ended_reads(paths):
            idx += 1
            assert count == 21
            assert len(names) == count
            assert len(reads) == count * 2
            for read in reads:
                assert set(read) <= _BASE_SET
        assert idx == 1


class TestMapper:

    def test_map_reads(self, shared_datadir, seekmer_index):
        paths = [shared_datadir / '20_1.fastq', shared_datadir / '20_2.fastq']
        map_result = mapper.map_reads(mapper.feed_pair_ended_reads(paths),
                                      seekmer_index)
        unaligned, class_map, class_count = map_result.summarize()
        assert unaligned == 0
