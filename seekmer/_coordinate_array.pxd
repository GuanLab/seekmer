# cython: language_level=3

"""Coordinate array functions"""

cimport libc.stdlib
import cython

from . cimport _coordinate


cdef struct CoordinateArray:
    int size
    _coordinate.Coordinate *items


cdef inline CoordinateArray empty_array() nogil:
    cdef CoordinateArray array
    array.size = 0
    array.items = NULL
    return array


cdef inline CoordinateArray create(int size) nogil:
    """Create an coordinate array of the specified size.

    Parameters
    ----------
    size : int
        The size

    Returns
    -------
    seekmer._coordinate_array.CoordinateArray
        A coordinate array of the specified size.
    """
    cdef CoordinateArray array
    array.size = size
    array.items = <_coordinate.Coordinate *>libc.stdlib.malloc(
        sizeof(_coordinate.Coordinate) * size
    )
    return array


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void reverse_complement(CoordinateArray array) nogil:
    """Reverse complement a coordinate list.

    Parameters
    ----------
    array : seekmer._coordinate_array.CoordinateArray
        A coordinate array.
    """
    cdef int size = array.size
    cdef _coordinate.Coordinate temp
    cdef int i
    for i in range(size // 2):
        temp = array.items[i]
        array.items[i] = array.items[size - i - 1]
        array.items[size - i - 1] = temp
    for i in range(size):
        array.items[i].entry = ~array.items[i].entry


cdef inline void free(CoordinateArray *array) nogil:
    libc.stdlib.free(array.items)
    array[0] = empty_array()


cdef inline bint is_empty(CoordinateArray targets) nogil:
    return targets.size == 0
