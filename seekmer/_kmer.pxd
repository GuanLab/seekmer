"""K-mer functions"""

cimport cpython.mem
cimport libc.stdint

import cython


cdef inline int size() nogil:
    """Return the hard-coded K-mer size

    Returns
    -------
    int
        The K-mer size
    """
    return 25


cdef inline libc.stdint.uint64_t get_invalid() nogil:
    """Return an invalid K-mer

    Returns
    -------
    libc.stdint.uint64_t
        The invalid K-mer
    """
    return 0xFFFFFFFFFFFFFFFFULL


cdef inline libc.stdint.uint64_t mask() nogil:
    """Return the K-mer mask

    Returns
    -------
    libc.stdint.uint64_t
        The bit-mask of K-mer
    """
    return ~(get_invalid() << (size() * 2))


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef inline libc.stdint.uint64_t encode(
        const char *sequence, int offset,
) nogil:
    """Encode the given sequence to a K-mer.

    Parameters
    ----------
    sequence : char *
        The sequence for encoding
    offset : int
        The starting point for encoding

    Returns
    -------
    libc.stdint.uint64_t
        The encoded K-mer
    """
    cdef libc.stdint.uint64_t kmer = 0
    cdef int i
    for i in range(offset, offset + size()):
        kmer <<= 2
        kmer |= _two_bit_encode(sequence[i])
    return kmer


cdef inline libc.stdint.uint64_t append(libc.stdint.uint64_t kmer,
                                        char base) nogil:
    """Append a base to the given K-mer.

    Parameters
    ----------
    kmer : libc.stdint.uint64_t
        The encoded K-mer
    base : char
        The starting point for encoding

    Returns
    -------
    libc.stdint.uint64_t
        The new K-mer
    """
    return ((kmer << 2) | _two_bit_encode(base)) & mask()


cdef inline libc.stdint.uint64_t prepend(libc.stdint.uint64_t kmer,
                                         char base) nogil:
    """Prepend a base to the given K-mer.

    Parameters
    ----------
    kmer : libc.stdint.uint64_t
        The encoded K-mer
    base : char
        The starting point for encoding

    Returns
    -------
    libc.stdint.uint64_t
        The new K-mer
    """
    return (kmer >> 2) | (_two_bit_encode(base) << (size() * 2 - 2))


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef inline bytes decode(libc.stdint.uint64_t kmer):
    """Encode the given sequence to a K-mer.

    Parameters
    ----------
    kmer : libc.stdint.uint64_t
        A given K-mer

    Returns
    -------
    bytes
        The decoded sequence bytestring
    """
    cdef char *sequence = <char *>cpython.mem.PyMem_Malloc(size())
    cdef int i
    cdef libc.stdint.uint64_t base
    for i in range(size() - 1, -1, -1):
        base = kmer & 0x3
        if base == 0:
            sequence[i] = b'A'
        elif base == 1:
            sequence[i] = b'C'
        elif base == 2:
            sequence[i] = b'G'
        else:
            sequence[i] = b'T'
        kmer >>= 2
    try:
        return sequence[:size()]
    finally:
        cpython.mem.PyMem_Free(sequence)


cdef inline libc.stdint.uint64_t reverse_complement(
        libc.stdint.uint64_t kmer
) nogil:
    """Reverse-complement a K-mer.

    Parameters
    ----------
    kmer : libc.stdint.uint64_t
        A given K-mer

    Returns
    -------
    libc.stdint.uint64_t
        The reverse-complemented K-mer
    """
    kmer = (((kmer >> 2) & 0x3333333333333333ULL)
            | ((kmer & 0x3333333333333333ULL) << 2))
    kmer = (((kmer >> 4) & 0x0f0f0f0f0f0f0f0fULL)
            | ((kmer & 0x0f0f0f0f0f0f0f0fULL) << 4))
    kmer = (((kmer >> 8) & 0x00ff00ff00ff00ffULL)
            | ((kmer & 0x00ff00ff00ff00ffULL) << 8))
    kmer = (((kmer >> 16) & 0x0000ffff0000ffffULL)
            | ((kmer & 0x0000ffff0000ffffULL) << 16))
    kmer = (kmer >> 32) | (kmer << 32)
    kmer = kmer >> (64 - size() * 2)
    return ~kmer & mask()


cdef inline int hash(libc.stdint.uint64_t kmer) nogil:
    """Hash a K-mer using the SipHash24 algorithm.

    Parameters
    ----------
    kmer : libc.stdint.uint64_t
        A given K-mer

    Returns
    -------
    int
        The hash value
    """
    ## The input sequence is under users' control, so the hash can use a
    ## fixed key.
    cdef libc.stdint.uint64_t k0 = 5381
    cdef libc.stdint.uint64_t k1 = 42
    cdef libc.stdint.uint64_t b = 8ULL << 56
    cdef libc.stdint.uint64_t v0 = k0 ^ 0x736f6d6570736575ULL
    cdef libc.stdint.uint64_t v1 = k1 ^ 0x646f72616e646f6dULL
    cdef libc.stdint.uint64_t v2 = k0 ^ 0x6c7967656e657261ULL
    cdef libc.stdint.uint64_t v3 = k1 ^ 0x7465646279746573ULL
    cdef libc.stdint.uint64_t mi = kmer
    v3 ^= mi
    _sip_half_round(&v0, &v1, &v2, &v3, 13, 16)
    _sip_half_round(&v2, &v1, &v0, &v3, 17, 21)
    _sip_half_round(&v0, &v1, &v2, &v3, 13, 16)
    _sip_half_round(&v2, &v1, &v0, &v3, 17, 21)
    v0 ^= mi
    b |= 0
    v3 ^= b
    _sip_half_round(&v0, &v1, &v2, &v3, 13, 16)
    _sip_half_round(&v2, &v1, &v0, &v3, 17, 21)
    _sip_half_round(&v0, &v1, &v2, &v3, 13, 16)
    _sip_half_round(&v2, &v1, &v0, &v3, 17, 21)
    v0 ^= 0
    v2 ^= 0xff
    _sip_half_round(&v0, &v1, &v2, &v3, 13, 16)
    _sip_half_round(&v2, &v1, &v0, &v3, 17, 21)
    _sip_half_round(&v0, &v1, &v2, &v3, 13, 16)
    _sip_half_round(&v2, &v1, &v0, &v3, 17, 21)
    _sip_half_round(&v0, &v1, &v2, &v3, 13, 16)
    _sip_half_round(&v2, &v1, &v0, &v3, 17, 21)
    _sip_half_round(&v0, &v1, &v2, &v3, 13, 16)
    _sip_half_round(&v2, &v1, &v0, &v3, 17, 21)
    return <int>((v0 ^ v1) ^ (v2 ^ v3))


cdef inline void _sip_half_round(
        libc.stdint.uint64_t *a, libc.stdint.uint64_t *b,
        libc.stdint.uint64_t *c, libc.stdint.uint64_t *d, int s, int t
) nogil:
    """The half round routine in SipHash24."""
    a[0] += b[0]
    c[0] += d[0]
    b[0] = ((b[0] << s) | (b[0] >> (64 - s))) ^ a[0]
    d[0] = ((d[0] << t) | (d[0] >> (64 - t))) ^ c[0]
    a[0] = ((a[0] << 32) | (a[0] >> 32))


cdef inline bint is_valid(libc.stdint.uint64_t kmer) nogil:
    """Check whether a K-mer is valid.

    If the K-mer equals to `seekmer._kmer.invalid()`, The K-mer is
    invalid.

    Parameters
    ----------
    kmer : libc.stdint.uint64_t
        A given K-mer

    Returns
    -------
    bool
        Whether the K-mer is valid
    """
    return kmer != get_invalid()


cdef inline libc.stdint.uint64_t _two_bit_encode(char base) nogil:
    """Two-bit encode a base.

    Parameters
    ----------
    base : char
        A given base

    Returns
    -------
    libc.stdint.uint64_t
        Whether the K-mer is valid
    """
    if base == ord(b'T') or base == ord(b't'):
        return 3
    elif base == ord(b'G') or base == ord(b'g'):
        return 2
    elif base == ord(b'C') or base == ord(b'c'):
        return 1
    else:
        return 0
