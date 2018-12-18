# cython: language_level=3

cimport libc.stdlib
cimport cython


cdef struct Sequence:
    int length
    char *bases


cdef inline Sequence create(int length) nogil:
    """Create a sequence of the given length

    Parameters
    ----------
    length : int
        The length of the sequences.

    Returns
    -------
    seekmer._sequence.Sequence
        An empty sequence.
    """
    cdef Sequence sequence
    sequence.length = length
    sequence.bases = <char *>libc.stdlib.malloc(sizeof(char) * (length + 1))
    sequence.bases[length] = 0
    return sequence


cdef inline Sequence from_bytes(bytes source):
    """Copy a sequence from bytes.

    Parameters
    ----------
    source : bytes
        The source sequence.

    Returns
    -------
    seekmer._sequence.Sequence
        A sequence.
    """
    cdef Sequence sequence
    sequence.length = len(source)
    sequence.bases = source
    return sequence


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void reverse_complement(Sequence sequence) nogil:
    """Reverse complement a sequence.

    Parameters
    ----------
    sequence : seekmer._sequence.Sequence
        A nucleotide sequence.
    """
    cdef char temp
    cdef int i
    for i in range(sequence.length // 2):
        temp = sequence.bases[sequence.length - i - 1]
        sequence.bases[sequence.length - i - 1] = sequence.bases[i]
        sequence.bases[i] = temp
    for i in range(sequence.length):
        if sequence.bases[i] == b'A':
            sequence.bases[i] = b'T'
        elif sequence.bases[i] == b'T':
            sequence.bases[i] = b'A'
        elif sequence.bases[i] == b'C':
            sequence.bases[i] = b'G'
        elif sequence.bases[i] == b'G':
            sequence.bases[i] = b'C'


cdef inline void free(Sequence *sequence) nogil:
    sequence.length = 0
    libc.stdlib.free(sequence.bases)
    sequence.bases = NULL
