__all__ = ('quantify', )

import warnings

import logbook
import numpy
import scipy.stats

_LOG = logbook.Logger(__name__)


def quantify(index, results, x0=None, bootstrap=False):
    """Estimate the transcript abundance.

    Parameters
    ----------
    index : seekmer.KMerIndex
        The Seekmer index.
    results : seekmer.SummarizedResults
    x0 : numpy.ndarray[float]
        The initial guess of the abundance.
    bootstrap : bool
        Whether to bootstrap reads before quantification.

    Returns
    -------
    numpy.ndarray
        The expression level of the transcripts.
    """
    if not bootstrap:
        _LOG.info('Aligned {} reads ({:.2%})', results.aligned,
                  results.aligned / results.total)
    if results.class_map.size == 0:
        return numpy.zeros(index.transcripts.size, dtype='f4')
    if bootstrap:
        n = results.class_count.sum()
        p = results.class_count.astype('f8') / n
        class_count = scipy.stats.multinomial(n=n, p=p).rvs().flatten()
    else:
        class_count = results.class_count
    # NOTE: The lengths are intentionally not adjusted. Current tests showed
    # the traditional correction using read fragment length distribution gives
    # higher errors in the final estimation.
    transcript_length = index.transcripts['length'].astype('f4')
    if x0 is None:
        x = numpy.ones(transcript_length.size, dtype='f4') / transcript_length
    else:
        x = x0
    x /= x.sum()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', 'divide by zero encountered in true_divide',
        )
        warnings.filterwarnings(
            'ignore', 'invalid value encountered in true_divide',
        )
        x = em(x, transcript_length, results.class_map, class_count)
    x /= x.sum() / 1000000
    x[x < 0.001] = 0
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
    x[x != x] = 0
    while (numpy.absolute(x - old_x) / x)[x > 1e-8].max() > 0.01:
        old_x = x
        x = x[class_map[1]]
        class_inner = numpy.bincount(class_map[0], weights=x,
                                     minlength=class_count.size) / class_count
        x = numpy.bincount(class_map[1], weights=x / class_inner[class_map[0]],
                           minlength=l.size) / l / n
        x[x != x] = 0
    return x
