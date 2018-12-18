"""Coordinate functions"""

cimport libc.stdint
import cython


## 64-bit encoded coordinate
cdef struct Coordinate:
    libc.stdint.int32_t entry
    libc.stdint.int32_t offset


cdef inline Coordinate get_invalid() nogil:
    """Get an invalid coordinate.

    Returns
    -------
    seekmer._coordinate.Coordinate
        An invalid coordinate
    """
    cdef Coordinate coordinate
    coordinate.entry = 0
    coordinate.offset = -1
    return coordinate


cdef inline libc.stdint.int64_t encode(Coordinate coordinate) nogil:
    """Encode an coordinate to a 64-bit integer.

    Parameters
    ----------
    coordinate : seekmer._coordinate.Coordinate
        A coordinate

    Returns
    -------
    libc.stdint.int64_t
        A 64-bit integer encoding the given coordinate
    """
    return <libc.stdint.int64_t> (
            (<libc.stdint.uint64_t> coordinate.entry << 32)
            | <libc.stdint.uint64_t> coordinate.offset
    )


cdef inline Coordinate decode(libc.stdint.int64_t value) nogil:
    """Decode a 64-bit integer to an coordinate

    Parameters
    ----------
    value : libc.stdint.int64_t
        A 64-bit integer encoding a coordinate

    Returns
    -------
    seekmer._coordinate.Coordinate
        The decoded coordinate
    """
    cdef Coordinate result
    result.entry = <int> (value >> 32)
    result.offset = <int> value
    return result


cdef inline Coordinate reverse_complement(Coordinate coordinate) nogil:
    """Reverse-complement a coordinate.

    Parameters
    ----------
    coordinate : seekmer._coordinate.Coordinate
        A coordinate

    Returns
    -------
    seekmer._coordinate.Coordinate
        The coordinate with a bitwise negated entry index.
    """
    cdef Coordinate result
    result.entry = ~coordinate.entry
    result.offset = coordinate.offset
    return result


cdef inline bint is_valid(Coordinate value) nogil:
    """Check whether a coordinate is valid.

    A coordinate with a negative offset is invalid.

    Parameters
    ----------
    coordinate : seekmer._coordinate.Coordinate
        A coordinate

    Returns
    -------
    bool
        Whether a coordinate is valid
    """
    return value.offset >= 0


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef inline int compare(const void *item1, const void *item2) nogil:
    """Compare two coordinates.

    Parameters
    ----------
    item1 : seekmer._coordinate.Coordinate *
        A pointer to a coordinate
    item2 : seekmer._coordinate.Coordinate *
        A pointer to a coordinate

    Returns
    -------
    int
        Return 0 if two coordinates are the same. Return -1 if the first
        coordinate is smaller than the second one. Return 1 if the first
        coordinate is larger than the second one.
    """
    cdef Coordinate coordinate1 = (<Coordinate *>item1)[0]
    cdef Coordinate coordinate2 = (<Coordinate *>item2)[0]
    if coordinate1.entry < coordinate2.entry:
        return -1
    elif coordinate1.entry > coordinate2.entry:
        return 1
    elif coordinate1.offset < coordinate2.offset:
        return -1
    elif coordinate1.offset > coordinate2.offset:
        return 1
    else:
        return 0
