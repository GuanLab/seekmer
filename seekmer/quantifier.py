__all__ = ('quantify')

import logbook
import numpy
import scipy.optimize


_LOG = logbook.Logger(__name__)


def quantify(index, class_map, class_count, mean_fragment_length, x=None):
    """Estimate the transcript abundance.

    Parameters
    ----------
    index : seekmer.KMerIndex
        The Seekmer index.
    class_map : numpy.ndarray
        An array of equivalent classes and their mappable targets.
    class_count : numpy.ndarray
        The number of reads in each class.
    mean_fragment_length: float
        The harmonic mean of the fragment lengths
    x: numpy.ndarray
        The initial guess of the abundance.

    Returns
    -------
    numpy.ndarray
        The expression level of the transcripts.
    """
    if class_map.size == 0:
        return numpy.zeros(index.transcripts.size, dtype='f4')
    if x is None:
        x = numpy.zeros(len(index.transcripts), dtype='f4')
    transcript_length = (index.transcripts['length'].astype('f4')
                         - mean_fragment_length).clip(min=1.0)
    convergence_check = _convergence_check(x, 50000)
    try:
        x = scipy.optimize.minimize(
            _score, x, args=(transcript_length, class_map, class_count),
            method='L-BFGS-B', jac=True, callback=convergence_check,
            tol=numpy.finfo('f8').eps,
        )['x']
    except StopIteration as e:
        x = e.args[0]
    x = numpy.exp(x - x.max())
    x /= x.sum() / 1000000
    x[x < 0.01] = 0.0
    x /= x.sum() / 1000000
    return x


def _score(x, l, class_map, class_count):
    """
    Compute the negative log likelihood function.

    Parameters
    ----------
    x: numpy.ndarray
        The transformed expression levels. Take softmax to get the
        original expression levels.
    l: numpy.ndarray
        All the transcript length. It should have the size of x.
    class_count : numpy.ndarray
        The read map array.
    class_map : numpy.ndarray
        The class-to-transcript annotation array of the read map.

    Returns
    -------
    float
        The negative log likelihood function
    numpy.ndarray
        The gradient of x
    """
    alpha = numpy.exp(x)
    alpha /= alpha.sum()
    alpha_over_l = alpha / l
    class_inner = numpy.bincount(
        class_map[0], weights=alpha_over_l[class_map[1]],
        minlength=class_count.size,
    )
    value = -(class_count * numpy.log(class_inner)).sum()
    gradient = alpha * class_count.sum() - alpha_over_l * numpy.bincount(
        class_map[1], weights=(class_count / class_inner)[class_map[0]],
        minlength=x.size,
    )
    return value, gradient


def _convergence_check(initial_x, max_round):
    round = 0
    def _wrapper(x):
        nonlocal initial_x, round
        round += 1
        if ((x < 1e-8) | (numpy.absolute(initial_x - x) / x < 1e-5)).all():
            _LOG.info('Converged after {} round(s) of iteration', round)
            _LOG.debug(x)
            _LOG.debug(numpy.absolute(initial_x - x))
            raise StopIteration(x)
        if round == max_round:
            _LOG.info('Maximum rounds of iteration ({}) reached.', round)
            raise Stopiteration(x)
        initial_x = x
    return _wrapper
