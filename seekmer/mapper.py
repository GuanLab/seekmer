import collections
import queue
import threading

import logbook
import numpy

from ._mapper import MAX_FRAGMENT_LENGTH, ReadMapper

__all__ = ('MAX_FRAGMENT_LENGTH', 'MapResult',
           'ReadMapper', 'SummarizedResult')

_LOG = logbook.Logger(__name__)

EPS = numpy.finfo('f4').eps

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

    def summarize(self):
        """Summarize the results.

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
            effective_lengths=self.effective_lengths,
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

    @property
    def effective_lengths(self):
        length = self.index.transcripts['length']
        effective_length = numpy.zeros(length.shape, dtype='f8')
        p = self.fragment_length_counts / self.fragment_length_counts.sum()
        for i in range(p.size):
            effective_length += (length - i).clip(min=1) * p[i]
        return effective_length

    def clear(self):
        """Clear the counter."""
        self.counter.clear()


def map_reads(index, read_feeder, job_count=1, readmap=None, debug=False):
    """Map reads.

    Parameters
    ----------
    index : seekmer.KMerIndex
        The K-mer index.
    read_feeder : iterator of reads
        The read feeder.
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
                thread = threading.Thread(
                    target=ReadMapper(index, map_result),
                    args=(iter(reads_queue.get, None),)
                )
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
