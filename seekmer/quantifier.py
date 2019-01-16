__all__ = ('quantify', )

import warnings

import logbook
import numpy

_LOG = logbook.Logger(__name__)


def quantify(index, class_map, class_count, x=None):
    """Estimate the transcript abundance.

    Parameters
    ----------
    index : seekmer.KMerIndex
        The Seekmer index.
    class_map : numpy.ndarray
        An array of equivalent classes and their mappable targets.
    class_count : numpy.ndarray
        The number of reads in each class.
    x: numpy.ndarray
        The initial guess of the abundance.

    Returns
    -------
    numpy.ndarray
        The expression level of the transcripts.
    """
    if class_map.size == 0:
        return numpy.zeros(index.transcripts.size, dtype='f4')
    transcript_length = index.transcripts['length'].astype('f4')
    if x is None:
        x = numpy.ones(len(index.transcripts), dtype='f4') / transcript_length
    x /= x.sum()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', 'divide by zero encountered in true_divide',
        )
        warnings.filterwarnings(
            'ignore', 'invalid value encountered in true_divide',
        )
        x = em(x, transcript_length, class_map, class_count)
    x /= x.sum() / 1000000
    return x


def em(x, l, class_map, class_count):
    """Expectation-maximization.

    Parameters
    ----------
    x: numpy.ndarray[float]
        The initial guess of the abundance.
    l: numpy.ndarray[float]
        The harmonic mean of the fragment lengths
    class_map : numpy.ndarray[int]
        An array of equivalent classes and their mappable targets.
    class_count : numpy.ndarray[int]
        The number of reads in each class.

    Returns
    -------
    numpy.ndarray[float]
        The expression level of the transcripts.
    """
    n = class_count.sum()
    old_x = x
    x = x[class_map[1]]
    class_inner = numpy.bincount(class_map[0], weights=x,
                                 minlength=class_count.size) / class_count
    x = numpy.bincount(class_map[1], weights=x / class_inner[class_map[0]],
                       minlength=l.size) / l / n
    while (numpy.absolute(x - old_x) / x)[x > 1e-8].max() > 0.01:
        old_x = x
        x = x[class_map[1]]
        class_inner = numpy.bincount(class_map[0], weights=x,
                                     minlength=class_count.size) / class_count
        x = numpy.bincount(class_map[1], weights=x / class_inner[class_map[0]],
                           minlength=l.size) / l / n
    return x
