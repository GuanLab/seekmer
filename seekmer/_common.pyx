"""Common data structures in Seekmer"""

__all__ = ('INVALID_INDEX', 'KMerIndex')

import tempfile

import logbook
import numpy
import tables
cimport cython

from . cimport _kmer


INVALID_INDEX = _INVALID_INDEX

_LOG = logbook.Logger(__name__)


cdef class KMerIndex:

    def __init__(self, kmers, contigs, sequences, targets, transcripts, exons):
        """Create a KMerIndex.

        Parameters
        ----------
        kmers : numpy.ndarray
            A K-mer table.
        contigs : numpy.ndarray
            A contig table.
        sequences : numpy.ndarray
            All contig sequences.
        targets : numpy.ndarray
            The contig map.
        transcripts : numpy.ndarray
            A transcript table.
        exons : numpy.ndarray
            An exon table.
        """
        self.kmers = kmers
        self.kmers_view = self.kmers
        self.contigs = contigs
        self.contigs_view = self.contigs
        self.targets = targets
        self.targets_view = self.targets
        self.sequences = sequences
        self.sequences_view = self.sequences
        self.transcripts = transcripts
        self.exons = exons

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef _coordinate.Coordinate map_kmer(
            self, libc.stdint.uint64_t kmer
    ) nogil:
        """Map a K-mer to the index.

        Identify the original contig to which the specified K-mer maps
        to. The contig index will be bitwise-negated if the
        reverse-complement K-mer maps to the contig.

        Parameters
        ----------
        kmer : libc.stdint.uint64_t
            A encoded K-mer

        Returns
        -------
        position : seekmer._coordinate.Coordinate
            The coordinate of the K-mer in contigs. The contig index is
            bitwise-negated if the reverse-complement K-mer maps to the
            index table.
        """
        cdef libc.stdint.uint64_t rc_kmer = _kmer.reverse_complement(kmer)
        cdef int size = self.kmers_view.shape[0]
        cdef int offset = _kmer.hash(min(kmer, rc_kmer)) & (size - 1)
        cdef int i
        for i in range(offset, size):
            if not _kmer.is_valid(self.kmers_view[i].kmer):
                return _coordinate.get_invalid()
            if self.kmers_view[i].kmer == kmer:
                return self.kmers_view[i].position
            if self.kmers_view[i].kmer == rc_kmer:
                return _coordinate.reverse_complement(
                    self.kmers_view[i].position
                )
        for i in range(offset):
            if not _kmer.is_valid(self.kmers_view[i].kmer):
                return _coordinate.get_invalid()
            if self.kmers_view[i].kmer == kmer:
                return self.kmers_view[i].position
            if self.kmers_view[i].kmer == rc_kmer:
                return _coordinate.reverse_complement(
                    self.kmers_view[i].position
                )
        return _coordinate.get_invalid()

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef _sequence.Sequence get_contig_sequence(
            self, _coordinate.Coordinate coordinate, int length,
    ) nogil:
        """Fetch the specified contig sequence.

        Parameters
        ----------
        coordinate : seekmer._coordinate.Coordinate
            The contig coordinate.
        length : int
            The sequence length. If the number is negative, the contig
            is aligned to the right edge of the corresponding K-mer.

        Returns
        -------
        seekmer._sequence.Sequence
            The specified sequence
        """
        cdef int index = coordinate.entry
        if index < 0:
            index = ~index
        cdef int offset = self.contigs_view[index].offset + coordinate.offset
        if coordinate.entry >= 0:
            offset += length if length > 0 else _kmer.size()
        else:
            offset += _kmer.size() if length > 0 else -length
        if length < 0:
            length = -length
        cdef _sequence.Sequence sequence = _sequence.create(length)
        cdef int i
        for i in range(offset - length, offset):
            sequence.bases[i - offset + length] = self.sequences_view[i]
        if coordinate.entry < 0:
            _sequence.reverse_complement(sequence)
        return sequence

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef _coordinate_array.CoordinateArray map_contig(
            self, _coordinate.Coordinate coordinate
    ) nogil:
        """Map a contig to the transcripts.

        Identify the original transcripts to which the specified contig
        can maps to.

        Parameters
        ----------
        coordinate : seekmer._coordinate.Coordinate
            The given contig coordinate

        Returns
        -------
        target: _coordinate_array.CoordinateArray *
            A list of mappable transcript indices
        """
        cdef int index = coordinate.entry
        cdef bint forward = index >= 0
        if not forward:
            index = ~index
        cdef int start = self.contigs_view[index].target_offset
        cdef int length = self.contigs_view[index].target_length
        cdef _coordinate_array.CoordinateArray targets = (
                _coordinate_array.create(length)
        )
        cdef int i, j
        if forward:
            for i, j in enumerate(range(start, start + length)):
                targets.items[i] = self.targets_view[j]
        else:
            for i, j in enumerate(range(start + length - 1, start - 1, -1)):
                targets.items[i] = _coordinate.reverse_complement(
                    self.targets_view[j]
                )
        return targets

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef bint _filter_on_contig(self, MappedSpan *span) nogil:
        """Filter a target list based on the specified contig.

        If there is no common element found, the target list will be
        intact, and the function returns False.

        Parameters
        ----------
        span: MappedSpan
            The mapped span information.

        Returns
        -------
        bint
            Whether there are common elements left.
        """
        if span.targets.size == 0:
            return True
        cdef int contig_id = span.anchor.entry
        cdef bint forward = contig_id >= 0
        if not forward:
            contig_id = ~contig_id
        cdef int start = self.contigs_view[contig_id].target_offset
        cdef int length = self.contigs_view[contig_id].target_length
        cdef int read_index = 0
        cdef int write_index = 0
        cdef int track_index = start if forward else start + length - 1
        cdef int track_bound = start + length if forward else start - 1
        cdef int step = 1 if forward else -1
        cdef int target_entry, index_entry, i
        while read_index != span.targets.size and track_index != track_bound:
            target_entry = span.targets.items[read_index].entry
            index_entry = self.targets_view[track_index].entry
            if not forward:
                index_entry = ~index_entry
            if target_entry == index_entry:
                span.targets.items[write_index] = (
                    span.targets.items[read_index]
                )
                read_index += 1
                write_index += 1
                track_index += step
            elif target_entry < index_entry:
                read_index += 1
            elif target_entry > index_entry:
                track_index += step
        if write_index == 0:
            return False
        else:
            span.targets.size = write_index
            return True

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef libc.stdint.uint64_t get_tail_kmer(
            self, _coordinate.Coordinate coordinate,
    ) nogil:
        """Fetch the tail K-mer of a contig.

        Parameters
        ----------
        coordinate : seekmer._coordinate.Coordinate
            A contig coordinate

        Returns
        -------
        libc.stdint.uint64_t
            The corresponding K-mer
        """
        cdef int index = coordinate.entry
        if index < 0:
            index = ~index
        cdef libc.stdint.uint64_t kmer
        if coordinate.offset == 0:
            kmer = self.contigs_view[index].first_kmer
        else:
            kmer = self.contigs_view[index].last_kmer
        if coordinate.entry < 0:
            kmer = _kmer.reverse_complement(kmer)
        return kmer

    def save(self, path):
        """Save the index to a HDF5 file.

        Parameters
        ----------
        path : pathlib.Path
            The path to which the index is saved.
        """
        filters = tables.Filters(complib='blosc', complevel=9, fletcher32=True)
        with tables.open_file(str(path), 'w', filters=filters) as index_file:
            index_file.root._v_attrs['seekmer_version'] = '2019.0.0'
            index_file.create_table('/', 'kmers', obj=self.kmers)
            index_file.create_table('/', 'contigs', obj=self.contigs)
            index_file.create_array('/', 'sequences', obj=self.sequences)
            index_file.create_table('/', 'targets', obj=self.targets)
            index_file.create_table('/', 'transcripts', obj=self.transcripts)
            index_file.create_table('/', 'exons', obj=self.exons)
        _LOG.info('Saved index to "{}"', path)

    @classmethod
    def load(cls, path):
        """Load the index from a binary file.

        Parameters
        ----------
        path : pathlib.Path
            The index file to read.

        Returns
        -------
        seekmer.KMerIndex
            The loaded index
        """
        with tables.open_file(str(path), 'r') as index_file:
            version = index_file.root._v_attrs['seekmer_version']
            if version != '2019.0.0':
                raise RuntimeError('invalid index version.')
            kmers = index_file.get_node('/kmers').read()
            contigs = index_file.get_node('/contigs').read()
            sequences = index_file.get_node('/sequences').read()
            targets = index_file.get_node('/targets').read()
            transcripts = index_file.get_node('/transcripts').read()
            exons = index_file.get_node('/exons').read()
        _LOG.info('Loaded index from "{}"', path)
        return KMerIndex(kmers, contigs, sequences, targets, transcripts,
                         exons)
