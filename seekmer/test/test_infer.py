import pytest

from .. import common
from .. import mapper
from .. import index_builder
from .. import infer

_BASE_SET = set(b'ACTGNactg')


@pytest.fixture
def seekmer_index(shared_datadir):
    gtf_path = shared_datadir / 'human.ens90.21.gtf.gz'
    fasta_path = shared_datadir / 'human.cdna.21.fa.bz2'
    exome = index_builder.load_exome(gtf_path)
    ids, sequences = index_builder.read_transcripts(fasta_path)
    index = index_builder.build(ids, sequences, exome)
    return index


class TestInfer:
    def test_quantify(self, shared_datadir, seekmer_index):
        paths = [shared_datadir / '20_1.fastq', shared_datadir / '20_2.fastq']
        map_result = mapper.map_reads(seekmer_index,
                                      common.feed_pair_ended_reads(*paths))
        summarized = map_result.summarize()
        infer.quantify(summarized)

    def test_bootstrap(self, shared_datadir, seekmer_index):
        paths = [shared_datadir / '20_1.fastq', shared_datadir / '20_2.fastq']
        map_result = mapper.map_reads(seekmer_index,
                                      common.feed_pair_ended_reads(*paths))
        summarized = map_result.summarize()
        x0 = infer.quantify(summarized)
        infer.quantify(summarized, x0=x0, bootstrap=True)
