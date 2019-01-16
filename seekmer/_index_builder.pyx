cimport libc.stdint
import gc

cimport cython
import logbook
import numpy
import numpy.core.records

from . cimport _common
from . cimport _kmer


_LOG = logbook.Logger(__name__)

_COMPLEMENT_TABLE = bytes.maketrans(b'ATCGatcg', b'TAGCTAGC')
"""bytes : A translation table for base complement."""

cdef int _INITIAL_INDEX_SIZE = 1024
"""int : The initial index size."""

cdef int INVALID_INDEX = 0x7FFFFFFF
INVALID_INDEX = INVALID_INDEX
"""int : An invalid index in the K-mer table."""


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
def extract_sequences(targets, bytes sequence):
    """Extract transcript sequences from the chromosome sequence.

    Parameters
    ----------
    targets : pandas.DataFrame
        A table of transcript exons.
    sequence : bytes
        The chromosome sequence.

    Returns
    -------
    list[bytes]
        A list of transcript sequences.
    """
    cdef list transcripts = []
    cdef list sequences = []
    cdef list sequence_chunks = []
    cdef libc.stdint.int32_t[:] start = targets['start']
    cdef libc.stdint.int32_t[:] end = targets['end']
    strand = targets['strand']
    cdef int i
    old_transcript = None
    for i, transcript in enumerate(targets['transcript_id']):
        if transcript != old_transcript:
            if old_transcript is not None:
                transcripts.append(old_transcript)
                sequences.append(b''.join(sequence_chunks))
                sequence_chunks.clear()
            old_transcript = transcript
        chunk = sequence[start[i]:end[i]]
        if not strand[i]:
            chunk = _reverse_complement(chunk)
        sequence_chunks.append(chunk)
    if old_transcript is not None:
        transcripts.append(old_transcript)
        sequences.append(b''.join(sequence_chunks))
    return transcripts, sequences


cdef bytes _reverse_complement(bytes sequence):
    return sequence.translate(_COMPLEMENT_TABLE)[::-1]


cdef struct LastKMerInfo:
    bint is_new
    bint forward
    int index


cdef struct GraphNodeEntry:
    libc.stdint.uint64_t kmer
    int last
    int next


cdef class ContigAssembler:
    """A K-mer index builder using de Bruijn Graph."""

    # The class follows a builder pattern. It is not implemented as a function
    # because assembly requires several internal states that are better
    # implemented as class fields.

    cdef LastKMerInfo last_kmer
    cdef int kmer_count
    cdef object kmer_table
    cdef GraphNodeEntry[::1] kmer_view

    def __init__(self):
        """Create a ContigAssembler."""
        self.last_kmer.is_new = True
        self.last_kmer.forward = True
        self.last_kmer.index = _common._INVALID_INDEX
        self.kmer_count = 0
        self.kmer_table = None

    def assemble(self, sequences):
        """Build a K-mer index.

        Parameters
        ----------
        sequences : list of bytes
            The transcript sequences.

        Returns
        -------
        kmer_table : numpy.recarray
            A numpy recarray that includes all K-mers and their contig
            mappings.
        contigs : numpy.recarray
            A numpy recarray that includes all contigs, their sequence
            offsets, and their targets mapping offsets.
        sequences : numpy.ndarray
            A numpy array of bytes includes pooled contig sequences.
        targets : numpy.ndarray
            A numpy array that includes the mapping data between contigs
            and transcripts.
        """
        gc.collect()
        self.kmer_count = 0
        self.kmer_table = numpy.ndarray(_INITIAL_INDEX_SIZE, dtype=[
            ('kmer', 'u8'), ('last', 'i4'), ('next', 'i4'),
        ])
        self.kmer_table['kmer'] = _kmer.get_invalid()
        self.kmer_table['last'] = _common._INVALID_INDEX
        self.kmer_table['next'] = _common._INVALID_INDEX
        self.kmer_view = self.kmer_table
        self._scan_kmers(sequences)
        self._connect_kmers(sequences)
        contigs = self._assemble_contigs()
        kmer_table = self.kmer_table.view(dtype=[
            ('kmer', 'u8'), ('entry', 'i4'), ('offset', 'i4'),
        ])
        targets = self._map_contigs(contigs, sequences)
        contigs, sequences, targets = self._compile_contigs(contigs, targets)
        self.last_kmer.is_new = True
        self.last_kmer.forward = True
        self.last_kmer.index = _common._INVALID_INDEX
        self.kmer_count = 0
        self.kmer_table = None
        self.kmer_view = None
        return kmer_table, contigs, sequences, targets

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def _scan_kmers(self, sequences):
        """Collect K-mers from all transcript sequences."""
        gc.collect()
        _LOG.info('Collecting K-mers from transcripts')
        cdef libc.stdint.uint64_t kmer
        cdef int i, j
        for i, sequence in enumerate(sequences):
            if i % 10000 == 0:
                _LOG.debug('Collected K-mers from {} transcripts', i)
            if len(sequence) < _kmer.size():
                continue
            kmer = _kmer.encode(sequence, 0) >> 2
            for j in range(_kmer.size() - 1, len(sequence)):
                kmer = _kmer.append(kmer, sequence[j])
                self._add_kmer(kmer)
        _LOG.info('Collected {} K-mers', self.kmer_count)
        _LOG.info('K-mer table size: {}', self.kmer_table.size)
        self.kmer_table['kmer'] = _kmer.get_invalid()
        self.kmer_table['last'] = _common._INVALID_INDEX
        self.kmer_table['next'] = _common._INVALID_INDEX

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef int _add_kmer(self, libc.stdint.uint64_t kmer) nogil:
        """Add a K-mer in the K-mer table.

        Register the K-mer in the K-mer table.

        Parameters
        ----------
        kmer : libc.stdint.uint64_t
            An encoded K-mer.
        """
        cdef int i = self.find_slot(kmer)
        if self.kmer_view[i].kmer == _kmer.get_invalid():
            self.kmer_count += 1
            if self.kmer_count > 0.8 * self.kmer_view.shape[0]:
                with gil:
                    self._expand()
                i = self.find_slot(kmer)
            self.kmer_view[i].kmer = kmer

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void _expand(self):
        """Expand the K-mer table by 2."""
        self.kmer_view = None
        _LOG.debug('K-mer count: {}', self.kmer_count)
        _LOG.debug('Expand the hash table: {}', self.kmer_table.size << 1)
        cdef int old_size = self.kmer_table.size
        self.kmer_table.resize(old_size << 1)
        self.kmer_table[old_size:]['kmer'] = _kmer.get_invalid()
        self.kmer_table[old_size:]['last'] = _common._INVALID_INDEX
        self.kmer_table[old_size:]['next'] = _common._INVALID_INDEX
        self.kmer_view = self.kmer_table
        cdef int i, j
        cdef libc.stdint.uint64_t kmer
        for i in range(old_size):
            if self.kmer_view[i].kmer == _kmer.get_invalid():
                continue
            kmer = self.kmer_view[i].kmer
            j = self.find_slot(kmer)
            if i != j:
                self.kmer_view[i].kmer = _kmer.get_invalid()
                self.kmer_view[j].kmer = kmer

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def _connect_kmers(self, sequences):
        """Connect K-mers into a de Bruijn graph."""
        gc.collect()
        _LOG.info('Connecting K-mers')
        cdef libc.stdint.uint64_t kmer
        cdef int i, j
        for i, sequence in enumerate(sequences):
            if i % 10000 == 0:
                _LOG.debug('Connecting K-mers in {} transcripts', i)
            if len(sequence) < _kmer.size():
                continue
            self.last_kmer.is_new = True
            self.last_kmer.forward = True
            self.last_kmer.index = _common._INVALID_INDEX
            kmer = _kmer.encode(sequence, 0) >> 2
            for j in range(_kmer.size() - 1, len(sequence)):
                kmer = _kmer.append(kmer, sequence[j])
                self._register_kmer(kmer)
            if self.last_kmer.index != _common._INVALID_INDEX:
                self._unlink(self.last_kmer.index, self.last_kmer.forward)
        _LOG.info('Connected all K-mers')

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void _register_kmer(self, libc.stdint.uint64_t kmer) nogil:
        """Register a K-mer in the K-mer table.

        Register the K-mer in the K-mer table. Connect or disconnect the
        K-mer based on its connectivity.

        Parameters
        ----------
        kmer : libc.stdint.uint64_t
            An encoded K-mer.
        """
        cdef libc.stdint.uint64_t rc_kmer = _kmer.reverse_complement(kmer)
        cdef libc.stdint.uint64_t index_kmer = min(kmer, rc_kmer)
        cdef int i = self.find_slot(index_kmer)
        cdef bint forward = index_kmer == kmer
        if self.last_kmer.index == _common._INVALID_INDEX:
            if _kmer.is_valid(self.kmer_view[i].kmer):
                self._unlink(i, not forward)
                self.last_kmer.is_new = False
            else:
                self.kmer_view[i].kmer = index_kmer
                self.last_kmer.is_new = True
            self.last_kmer.forward = forward
            self.last_kmer.index = i
            return
        if not _kmer.is_valid(self.kmer_view[i].kmer):
            self.kmer_view[i].kmer = index_kmer
            if self.last_kmer.is_new:
                self._link(self.last_kmer.index, i, self.last_kmer.forward)
                self._link(i, self.last_kmer.index, not forward)
            else:
                self._unlink(self.last_kmer.index, self.last_kmer.forward)
            self.last_kmer.is_new = True
            self.last_kmer.forward = forward
            self.last_kmer.index = i
            return
        if (self._get_link(self.last_kmer.index, self.last_kmer.forward) == i
                and self._get_link(i, not forward) == self.last_kmer.index):
            self.last_kmer.is_new = False
            self.last_kmer.forward = forward
            self.last_kmer.index = i
            return
        self._unlink(self.last_kmer.index, self.last_kmer.forward)
        self._unlink(i, not forward)
        if self._get_link(i, forward) != _common._INVALID_INDEX:
            self.last_kmer.is_new = False
            self.last_kmer.forward = forward
            self.last_kmer.index = i
        else:
            self.last_kmer.is_new = True
            self.last_kmer.forward = True
            self.last_kmer.index = _common._INVALID_INDEX

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef int find_slot(self, libc.stdint.uint64_t kmer) nogil:
        """Find a slot for the given K-mer

        Parameters
        ----------
        kmer : libc.stdint.uint64_t
            An encoded K-mer.

        Returns
        -------
        int
            The index of the slot for the given K-mer, or -1 if there
            is not one.
        """
        cdef libc.stdint.uint64_t rc_kmer = _kmer.reverse_complement(kmer)
        cdef libc.stdint.uint64_t index_kmer = min(kmer, rc_kmer)
        cdef int mask = self.kmer_view.shape[0] - 1
        cdef int offset = _kmer.hash(index_kmer) & mask
        cdef int i
        for i in range(offset, self.kmer_view.shape[0]):
            if (not _kmer.is_valid(self.kmer_view[i].kmer)
                    or self.kmer_view[i].kmer == kmer
                    or self.kmer_view[i].kmer == rc_kmer):
                return i
        for i in range(offset):
            if (not _kmer.is_valid(self.kmer_view[i].kmer)
                    or self.kmer_view[i].kmer == kmer
                    or self.kmer_view[i].kmer == rc_kmer):
                return i
        return -1

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void _link(self, int i, int j, bint is_forward) nogil:
        """Connect the specified K-mer.

        Connect the K-mer at the slot `i` to the K-mer at the slot `j`
        along the forward direction if `is_forward` is `True`, or
        backward direction if `is_forward` is `False`.

        Parameters
        ----------
        i : int
            The slot index of the origin K-mer
        j : int
            The slot index of the target K-mer
        is_forward : bint
            Whether the target K-mer is following the origin K-mer.
        """
        if is_forward:
            self.kmer_view[i].next = j
        else:
            self.kmer_view[i].last = j

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void _unlink(self, int i, bint is_forward) nogil:
        """Disconnect the specified K-mer.

        Disconnect the K-mer at the slot `i` along the forward direction
        if `is_forward` is `True`, or backward direction if `is_forward`
        is `False`. Disconnect the connecting K-mer as well.

        Parameters
        ----------
        i : int
            The slot index of the origin K-mer
        j : int
            The slot index of the target K-mer
        is_forward : bint
            Whether the target K-mer is following the origin K-mer.
        """
        cdef int j
        if is_forward:
            j = self.kmer_view[i].next
            self.kmer_view[i].next = _common._INVALID_INDEX
        else:
            j = self.kmer_view[i].last
            self.kmer_view[i].last = _common._INVALID_INDEX
        if j == _common._INVALID_INDEX:
            return
        if self.kmer_view[j].next == i:
            self.kmer_view[j].next = _common._INVALID_INDEX
        else:
            self.kmer_view[j].last = _common._INVALID_INDEX

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef int _get_link(self, int i, bint is_forward) nogil:
        """Fetch the slot index of a connected K-mer.

        Parameters
        ----------
        i : int
            The slot index of the origin K-mer.
        is_forward : bint
            Whether to fetch the following K-mer or the previous one.

        Returns
        -------
        int
            The slot index of the connected K-mer.
        """
        return self.kmer_view[i].next if is_forward else self.kmer_view[i].last

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def  _assemble_contigs(self):
        """Assemble K-mers into contigs."""
        gc.collect()
        contigs = []
        _LOG.info('Assembling contigs from K-mers')
        sequence_chunks = []
        cdef int i, last_index, index, offset
        cdef int contig_index = len(contigs)
        cdef _common.IndexEntry[:] kmer_index = self.kmer_table
        for i in range(kmer_index.shape[0]):
            # Skip empty slots.
            if not _kmer.is_valid(self.kmer_view[i].kmer):
                continue
            # Skip processed.
            if self.kmer_view[i].last < 0:
                continue
            # Skip if it is in the middle of a contig.
            if (self.kmer_view[i].last != _common._INVALID_INDEX
                    and self.kmer_view[i].next != _common._INVALID_INDEX):
                continue
            # Process a single K-mer contig.
            if (self.kmer_view[i].last == _common._INVALID_INDEX
                    and self.kmer_view[i].next == _common._INVALID_INDEX):
                kmer_index[i].position.entry = ~contig_index
                kmer_index[i].position.offset = ~0
                contigs.append(_kmer.decode(kmer_index[i].kmer))
                contig_index = len(contigs)
                continue
            # Process (an end of) a contig of multiple K-mers.
            sequence_chunks.clear()
            last_index = _common._INVALID_INDEX
            index = i
            offset = 0
            while index != _common._INVALID_INDEX:
                if self.kmer_view[index].next == last_index:
                    self.kmer_view[index].kmer = _kmer.reverse_complement(
                        self.kmer_view[index].kmer
                    )
                    self.kmer_view[index].last ^= self.kmer_view[index].next
                    self.kmer_view[index].next ^= self.kmer_view[index].last
                    self.kmer_view[index].last ^= self.kmer_view[index].next
                last_index = index
                index = self.kmer_view[index].next
                kmer_index[last_index].position.entry = ~contig_index
                kmer_index[last_index].position.offset = ~offset
                if offset % _kmer.size() == 0:
                    sequence_chunks.append(
                        _kmer.decode(kmer_index[last_index].kmer)
                    )
                offset += 1
            tail = _kmer.decode(kmer_index[last_index].kmer)
            offset = _kmer.size() - (offset - 1) % _kmer.size()
            sequence_chunks.append(tail[offset:])
            contigs.append(b''.join(sequence_chunks))
            contig_index = len(contigs)
        for i in range(kmer_index.shape[0]):
            kmer_index[i].position.entry = ~kmer_index[i].position.entry
            kmer_index[i].position.offset = ~kmer_index[i].position.offset
        _LOG.info('Assembled {} contigs.', len(contigs))
        return contigs

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def _map_contigs(self, contigs, sequences):
        """Map contigs back to transcripts."""
        gc.collect()
        _LOG.info('Mapping contigs')
        target_chunks = []
        targets = []
        cdef _common.IndexEntry[:] kmer_index = self.kmer_table
        cdef int i, j, k, contig_index, entry, offset
        cdef libc.stdint.uint64_t kmer
        for i, sequence in enumerate(sequences):
            if i % 10000 == 0:
                _LOG.debug('Mapped contigs to {} transcripts', i)
            if len(sequence) < _kmer.size():
                continue
            kmer = _kmer.encode(sequence, 0) >> 2
            for j in range(_kmer.size() - 1, len(sequence)):
                kmer = _kmer.append(kmer, sequence[j])
                k = self.find_slot(kmer)
                if kmer_index[k].position.offset != 0:
                    continue
                contig_index = kmer_index[k].position.entry
                entry = i
                offset = j - _kmer.size() + 1
                if kmer != kmer_index[k].kmer:
                    entry = ~entry
                targets.append((contig_index, entry, offset))
                if len(targets) > 4096:
                    target_chunks.append(numpy.core.records.fromrecords(
                        targets,
                        dtype=[
                            ('contig', 'i4'),
                            ('entry', 'i4'),
                            ('offset', 'i4'),
                        ]
                    ))
                    targets.clear()
        if targets:
            target_chunks.append(numpy.core.records.fromrecords(
                targets,
                dtype=[
                    ('contig', 'i4'),
                    ('entry', 'i4'),
                    ('offset', 'i4'),
                ]
            ))
        targets = numpy.concatenate(target_chunks)
        target_chunks.clear()
        targets.sort()
        _LOG.info('Mapped contigs')
        return targets

    def _compile_contigs(self, contigs, targets):
        """Pool all the contig sequences together.

        Reformat contigs information as a table. The table consists of
        their sequence offset information and their terminal K-mers.
        Drop all transcript sequences and only keep their lengths.
        """
        table = numpy.recarray(len(contigs), dtype=[
            ('offset', 'i8'),
            ('length', 'i8'),
            ('first_kmer', 'u8'),
            ('last_kmer', 'u8'),
            ('target_offset', 'i8'),
            ('target_count', 'i8'),
        ])
        table['length'] = [len(c) for c in contigs]
        table['offset'] = table['length'].cumsum() - table['length']
        table['target_count'] = numpy.bincount(targets['contig'],
                                               minlength=len(contigs))
        table['target_offset'] = (table['target_count'].cumsum()
                                  - table['target_count'])
        table['first_kmer'] = [_kmer.encode(contig, 0) for contig in contigs]
        table['last_kmer'] = [_kmer.encode(contig, len(contig) - _kmer.size())
                              for contig in contigs]
        sequences = numpy.frombuffer(bytearray(b''.join(contigs)), dtype='S1')
        contigs.clear()
        targets = numpy.copy(targets.getfield([('entry', 'i4'),
                                               ('offset', 'i4')], 4))
        return table, sequences, targets
