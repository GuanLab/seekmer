"""Common index-related definitions"""

cimport libc.stdint

from . cimport _coordinate
from . cimport _coordinate_array
from . cimport _sequence


cdef int _INVALID_INDEX = 0x7FFFFFFF
"""int : An invalid index in the K-mer table."""


## An single entry of the index table
cdef struct IndexEntry:
    libc.stdint.uint64_t kmer
    _coordinate.Coordinate position


## An single contig entry
cdef struct ContigEntry:
    libc.stdint.int64_t offset
    libc.stdint.int64_t length
    libc.stdint.uint64_t first_kmer
    libc.stdint.uint64_t last_kmer
    libc.stdint.int64_t target_offset
    libc.stdint.int64_t target_length


## A mapped segment in a read
cdef struct MappedSpan:
    libc.stdint.int32_t begin
    libc.stdint.int32_t end
    _coordinate.Coordinate anchor
    _coordinate_array.CoordinateArray targets


cdef class KMerIndex:
    """The core index of Seekmer

    Attributes
    ----------
    kmers : numpy.ndarray
        The K-mer index table
    contigs : numpy.ndarray
        The contig table
    sequences : numpy.ndarray
        The pooled contig sequences
    targets : numpy.ndarray
        The target table
    transcripts : numpy.ndarray
        The transcript table
    exons : numpy.ndarray
        The exon table
    """

    cdef public object kmers
    cdef IndexEntry[::1] kmers_view
    cdef public object contigs
    cdef ContigEntry[::1] contigs_view
    cdef public object targets
    cdef _coordinate.Coordinate[::1] targets_view
    cdef public object sequences
    cdef char[::1] sequences_view
    cdef public object transcripts
    cdef public object exons

    cdef _coordinate.Coordinate map_kmer(self, libc.stdint.uint64_t kmer) nogil
    cdef _sequence.Sequence get_contig_sequence(
            self, _coordinate.Coordinate coordinate, int length,
    ) nogil
    cdef _coordinate_array.CoordinateArray map_contig(
            self, _coordinate.Coordinate coordinate,
    ) nogil
    cdef bint _filter_on_contig(self, MappedSpan *span) nogil
    cdef libc.stdint.uint64_t get_tail_kmer(
            self, _coordinate.Coordinate coordinate,
    ) nogil
