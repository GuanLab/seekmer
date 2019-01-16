__all__ = ('MAX_FRAGMENT_LENGTH', 'ReadMapper')

cimport libc.stdlib

cimport cython
import logbook
import numpy

from . cimport _common
from . cimport _coordinate
from . cimport _coordinate_array
from . cimport _kmer
from . cimport _sequence


_LOG = logbook.Logger(__name__)

cdef int _MAX_FRAGMENT_LENGTH = 2000

MAX_FRAGMENT_LENGTH = _MAX_FRAGMENT_LENGTH

cdef int _ALIGN_LENGTH = 8

cdef int _MAX_OFFSET = 2

cdef int _MAX_DISTANCE = 4

cdef int _INVALID_SHIFT = 0x7FFF


cdef class ReadMapper:
    """A read mapper."""

    cdef _common.KMerIndex index
    cdef object map_result
    cdef object fragment_length_counts
    cdef libc.stdint.int64_t[:] fragment_length_counts_view

    def __init__(self, index, map_result):
        """Create a read mapper.

        Parameters
        ----------
        index : _common.KMerIndex
            The Seekmer index.
        map_result : seekmer.mapper.MapResult
            A mapping result collection.
        """
        self.index = index
        self.map_result = map_result
        self.fragment_length_counts = numpy.zeros(_MAX_FRAGMENT_LENGTH,
                                                  dtype='i8')
        self.fragment_length_counts_view = self.fragment_length_counts

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def __call__(self, reads_iterator):
        """Run the mapping loop.

        Repeatedly retrieve reads from the queue and map them. In the
        end, send the map back to the main process through the pipe.
        """
        cdef _coordinate_array.CoordinateArray *results = NULL
        cdef bint single_ended
        cdef int read_count
        cdef int i
        cdef int j
        cdef _sequence.Sequence *read_array
        cdef _common.MappedSpan span
        cdef int length
        for read_count, read_names, reads in reads_iterator:
            read_array = _build_read_array(reads)
            single_ended = read_count == len(reads)
            with nogil:
                libc.stdlib.free(results)
                results = _initialize_results(read_count)
                j = 0
                fragment_length = 0
                for i in range(read_count):
                    if single_ended:
                        span = self.map_read(read_array[j])
                        j += 1
                    else:
                        span = self.map_read_pair(read_array[j],
                                                  read_array[j + 1])
                        j += 2
                    results[i] = span.targets
                    length = span.end - span.begin + _kmer.size()
                    if length > 0:
                        if length >= _MAX_FRAGMENT_LENGTH:
                            length = _MAX_FRAGMENT_LENGTH - 1
                        self.fragment_length_counts_view[length] += 1
            result_ids = []
            for i in range(read_count):
                result_ids.append(_get_ids(results[i]))
                _coordinate_array.free(&results[i])
            with self.map_result.lock:
                self.map_result.update(read_names, result_ids)
            _LOG.debug('Mapped {} reads.', read_count)
        libc.stdlib.free(results)
        with self.map_result.lock:
            self.map_result.merge_fragment_lengths(self.fragment_length_counts)

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef _common.MappedSpan map_read_pair(self, _sequence.Sequence read1,
                                          _sequence.Sequence read2) nogil:
        """Map a read pair.

        Parameters
        ----------
        read1 : _sequence.Sequence
            A read.
        read2 : _sequence.Sequence
            A second read.

        Returns
        -------
        _common.MappedSpan
            A read mapping result
        """
        cdef _common.MappedSpan span1 = self.map_read(read1)
        cdef _common.MappedSpan span2 = self.map_read(read2)
        cdef int interval = 0
        if not _intersect(&span1, &span2):
            _coordinate_array.free(&span1.targets)
            span1.begin = 0
            span1.end = span1.begin - _kmer.size()
        elif span1.anchor.entry != ~span2.anchor.entry:
            span1.begin = 0
            span1.end = span1.begin - _kmer.size()
        else:
            span1.end = read1.length - _kmer.size()
            span2.end = read2.length - _kmer.size()
            interval = span2.anchor.offset - span1.anchor.offset
            if span1.anchor.entry < 0:
                interval = -interval
            span1.end += interval + span2.end - span2.begin
        _coordinate_array.free(&span2.targets)
        return span1

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef _common.MappedSpan map_read(self, _sequence.Sequence read) nogil:
        """Map a single read.

        Parameters
        ----------
        read : _sequence.Sequence
            A read.

        Returns
        -------
        _common.MappedSpan
            Read mapping status
        """
        cdef _common.MappedSpan span
        span.anchor = _coordinate.get_invalid()
        span.begin = 0
        span.end = span.begin
        span.targets = _coordinate_array.empty_array()
        self._find_first_kmer(read, &span)
        if _coordinate_array.is_empty(span.targets):
            return span
        if span.begin > 0:
            self._filter_targets_to_left(read, &span)
        if (not _coordinate_array.is_empty(span.targets)
                and span.end < read.length - _kmer.size()):
            self._filter_targets_to_right(read, &span)
        if not _coordinate_array.is_empty(span.targets):
            return span
        span.anchor = _coordinate.get_invalid()
        span.targets = _coordinate_array.empty_array()
        span.begin += _kmer.size()
        if span.begin + _kmer.size() > read.length:
            span.begin = read.length - _kmer.size()
        span.end = span.begin
        self._find_first_kmer(read, &span)
        if _coordinate_array.is_empty(span.targets):
            return span
        if span.begin > 0:
            self._filter_targets_to_left(read, &span)
        if (not _coordinate_array.is_empty(span.targets)
                and span.end < read.length - _kmer.size()):
            self._filter_targets_to_right(read, &span)
        return span

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void _find_first_kmer(self, _sequence.Sequence read,
                               _common.MappedSpan *span) nogil:
        cdef libc.stdint.uint64_t kmer = _kmer.encode(read.bases, span.begin)
        span.anchor = self.index.map_kmer(kmer)
        if span.anchor.offset >= 0:
            span.end = span.begin
            span.targets = self.index.map_contig(span.anchor)
            return
        cdef int i
        for i in range(span.begin + _kmer.size(), read.length):
            kmer = _kmer.append(kmer, read.bases[i])
            span.anchor = self.index.map_kmer(kmer)
            if span.anchor.offset < 0:
                continue
            span.begin = i + 1 - _kmer.size()
            span.end = span.begin
            span.targets = self.index.map_contig(span.anchor)
            return

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void _filter_targets_to_left(self, _sequence.Sequence read,
                                      _common.MappedSpan *span) nogil:
        cdef libc.stdint.uint64_t kmer = _kmer.encode(read.bases, span.begin)
        cdef bint forward = span.anchor.entry >= 0
        cdef libc.stdint.int32_t contig_index = (span.anchor.entry if forward
                                                 else ~span.anchor.entry)
        cdef int contig_length = self.index.contigs_view[contig_index].length
        cdef int move = span.anchor.offset if forward else (
                contig_length - span.anchor.offset - _kmer.size()
        )
        cdef _sequence.Sequence contig
        cdef int shift
        while span.begin > move:
            span.begin -= move
            span.anchor.offset -= move if forward else -move
            contig = self.index.get_contig_sequence(span.anchor, _ALIGN_LENGTH)
            shift = sift4_align_left(contig, read, span.begin)
            _sequence.free(&contig)
            if shift == _INVALID_SHIFT or shift + 1 + move <= 0:
                _coordinate_array.free(&span.targets)
                return
            span.begin -= shift + 1
            if span.begin < 0:
                span.begin = 0
                return
            kmer = _kmer.prepend(self.index.get_tail_kmer(span.anchor),
                                 read.bases[span.begin])
            span.anchor = self.index.map_kmer(kmer)
            if (not _coordinate.is_valid(span.anchor)
                    or not self.index._filter_on_contig(span)):
                if span.begin < _kmer.size():
                    # XXX: Should we count these bases in when we calculate
                    # the fragment length?
                    span.begin = 0
                    return
                span.begin -= _kmer.size()
                kmer = _kmer.encode(read.bases, span.begin)
                span.anchor = self.index.map_kmer(kmer)
                if (not _coordinate.is_valid(span.anchor)
                        or not self.index._filter_on_contig(span)):
                    _coordinate_array.free(&span.targets)
                    return
            forward = span.anchor.entry >= 0
            contig_index = span.anchor.entry if forward else ~span.anchor.entry
            contig_length = self.index.contigs_view[contig_index].length
            move = span.anchor.offset if forward else (
                contig_length - span.anchor.offset - _kmer.size()
            )
        span.anchor.offset -= span.begin if forward else -span.begin
        contig = self.index.get_contig_sequence(span.anchor, _ALIGN_LENGTH)
        shift = sift4_align_left(contig, read, 0)
        _sequence.free(&contig)
        if shift == _INVALID_SHIFT:
            _coordinate_array.free(&span.targets)

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void _filter_targets_to_right(self, _sequence.Sequence read,
                                       _common.MappedSpan *span) nogil:
        cdef libc.stdint.uint64_t kmer = _kmer.encode(read.bases, span.end)
        span.anchor = self.index.map_kmer(kmer)
        cdef bint forward = span.anchor.entry >= 0
        cdef libc.stdint.int32_t contig_index = (span.anchor.entry if forward
                                                 else ~span.anchor.entry)
        cdef int contig_length = self.index.contigs_view[contig_index].length
        cdef int move = ((contig_length - span.anchor.offset - _kmer.size())
                         if forward else span.anchor.offset)
        cdef _sequence.Sequence contig
        cdef int shift
        while read.length - span.end - _kmer.size() > move:
            span.end += move
            span.anchor.offset += move if forward else -move
            contig = self.index.get_contig_sequence(span.anchor,
                                                    -_ALIGN_LENGTH)
            shift = sift4_align_right(
                contig, read, span.end + _kmer.size() - _ALIGN_LENGTH,
            )
            _sequence.free(&contig)
            if shift == _INVALID_SHIFT or shift + 1 + move <= 0:
                _coordinate_array.free(&span.targets)
                return
            span.end += shift + 1
            if span.end + _kmer.size() > read.length:
                span.end = read.length - _kmer.size()
                return
            kmer = _kmer.append(self.index.get_tail_kmer(span.anchor),
                                read.bases[span.end + _kmer.size() - 1])
            span.anchor = self.index.map_kmer(kmer)
            if (not _coordinate.is_valid(span.anchor)
                    or not self.index._filter_on_contig(span)):
                _coordinate_array.free(&span.targets)
                return
            if (not _coordinate.is_valid(span.anchor)
                    or not self.index._filter_on_contig(span)):
                if span.end > read.length - 2 * _kmer.size():
                    # XXX: Should we count these bases in when we calculate
                    # the fragment length?
                    span.end = read.length - _kmer.size()
                    return
                span.end += _kmer.size()
                kmer = _kmer.encode(read.bases, span.end)
                span.anchor = self.index.map_kmer(kmer)
                if (not _coordinate.is_valid(span.anchor)
                        or not self.index._filter_on_contig(span)):
                    _coordinate_array.free(&span.targets)
                    return
            forward = span.anchor.entry >= 0
            contig_index = span.anchor.entry if forward else ~span.anchor.entry
            contig_length = self.index.contigs_view[contig_index].length
            move = ((contig_length - span.anchor.offset - _kmer.size())
                    if forward else span.anchor.offset)
        if forward:
            span.anchor.offset += read.length - span.end - _kmer.size()
        else:
            span.anchor.offset -= read.length - span.end - _kmer.size()
        contig = self.index.get_contig_sequence(span.anchor, -_ALIGN_LENGTH)
        shift = sift4_align_right(contig, read, read.length - _ALIGN_LENGTH)
        _sequence.free(&contig)
        if shift == _INVALID_SHIFT:
            _coordinate_array.free(&span.targets)


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef inline bint _intersect(_common.MappedSpan *result1,
                            _common.MappedSpan *result2) nogil:
    """Take the intersection of two result arrays.

    Take only the intersection of the coordinates from both the target
    and source array. The target array will be modified.

    Parameters
    ----------
    result1 : seekmer._coordinate_array.CoordinateArray *
        The target array of the first fragment.
    result2 : seekmer._coordinate_array.CoordinateArray *
        The target array of the second fragment.

    Returns
    -------
    bint
        Whether the target has any coordinates left. If the target is
        originally empty, it is always True.
    """
    if result1.targets.size == 0:
        return True
    if result2.targets.size == 0:
        return False
    cdef int cursor1_read = 0
    cdef int cursor1_write = 0
    cdef int cursor2 = result2.targets.size - 1
    cdef int entry1
    cdef int entry2
    while cursor1_read != result1.targets.size and cursor2 != -1:
        entry1 = result1.targets.items[cursor1_read].entry
        entry2 = ~result2.targets.items[cursor2].entry
        if entry1 == entry2:
            result1.targets.items[cursor1_write] = (
                result1.targets.items[cursor1_read]
            )
            cursor1_read += 1
            cursor1_write += 1
            cursor2 -= 1
        elif entry1 < entry2:
            cursor1_read += 1
        elif entry1 > entry2:
            cursor2 -= 1
    if cursor1_write == 0:
        return False
    else:
        result1.targets.size = cursor1_write
        return True


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int sift4_align_left(_sequence.Sequence reference,
                          _sequence.Sequence query, int offset) nogil:
    cdef int reference_cursor = reference.length - 1
    cdef int query_cursor = offset + _ALIGN_LENGTH - 1
    query_cursor -= 1
    cdef int distance = 0
    cdef int i
    while reference_cursor >= 0 and query_cursor >= offset:
        if _match_base(reference.bases[reference_cursor],
                       query.bases[query_cursor]):
            reference_cursor -= 1
            query_cursor -= 1
            continue
        if reference_cursor != query_cursor - offset:
            reference_cursor = min(query_cursor - offset, reference_cursor)
            query_cursor = reference_cursor + offset
        for i in range(_MAX_OFFSET):
            if (query_cursor - i >= offset - 1
                    and query_cursor - i >= 0
                    and _match_base(reference.bases[reference_cursor],
                                    query.bases[query_cursor - i])):
                distance += i - 1
                query_cursor -= i - 1
                reference_cursor += 1
                break
            if (reference_cursor - i >= 0
                    and _match_base(reference.bases[reference_cursor - i],
                                    query.bases[query_cursor])):
                distance += i - 1
                query_cursor += 1
                reference_cursor -= i - 1
                break
        distance += 1
        query_cursor -= 1
        reference_cursor -= 1
        if distance > _MAX_DISTANCE:
            return _INVALID_SHIFT
    if reference_cursor >= 0:
        return reference_cursor + 1
    if query_cursor >= offset:
        return -1 - query_cursor + offset
    return 0


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int sift4_align_right(_sequence.Sequence reference,
                           _sequence.Sequence query, int offset) nogil:
    cdef int reference_cursor = 0
    cdef int query_cursor = offset
    cdef int i = 0
    cdef int distance = 0
    while (reference_cursor < reference.length
           and query_cursor < offset + _ALIGN_LENGTH):
        if _match_base(reference.bases[reference_cursor],
                       query.bases[query_cursor]):
            reference_cursor += 1
            query_cursor += 1
            continue
        if reference_cursor != query_cursor - offset:
            reference_cursor = max(query_cursor - offset, reference_cursor)
            query_cursor = reference_cursor + offset
        for i in range(_MAX_OFFSET):
            if (query_cursor + i < offset + _ALIGN_LENGTH + 1
                    and query_cursor + i < query.length
                    and _match_base(reference.bases[reference_cursor],
                                    query.bases[query_cursor + i])):
                distance += i - 1
                query_cursor += i - 1
                reference_cursor -= 1
                break
            if (reference_cursor + i < reference.length
                    and _match_base(reference.bases[reference_cursor + i],
                                    query.bases[query_cursor])):
                distance += i - 1
                query_cursor -= 1
                reference_cursor += i - 1
                break
        distance += 1
        query_cursor += 1
        reference_cursor += 1
        if distance > _MAX_DISTANCE:
            return _INVALID_SHIFT
    if reference_cursor < reference.length:
        return reference.length - reference_cursor
    if query_cursor < offset + _ALIGN_LENGTH:
        return query_cursor - offset - _ALIGN_LENGTH
    return 0


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef bint _match_base(int reference, int query) nogil:
    return reference == query or query not in b'ACGT'


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef _sequence.Sequence *_build_read_array(list reads):
    cdef int i
    cdef _sequence.Sequence *sequences = <_sequence.Sequence *> (
            libc.stdlib.malloc(sizeof(_sequence.Sequence) * len(reads))
    )
    for i, read in enumerate(reads):
        sequences[i] = _sequence.from_bytes(read)
    return sequences


cdef _coordinate_array.CoordinateArray *_initialize_results(int size) nogil:
    return <_coordinate_array.CoordinateArray *> (
        libc.stdlib.malloc(sizeof(_coordinate_array.CoordinateArray) * size)
    )


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef tuple _get_ids(_coordinate_array.CoordinateArray array):
    cdef list ids = []
    cdef long i
    cdef libc.stdint.int32_t entry
    for i in range(array.size):
        entry = array.items[i].entry
        if entry < 0:
            entry = ~entry
        ids.append(entry)
    return tuple(ids)
