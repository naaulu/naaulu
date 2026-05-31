import matplotlib.pyplot
import numpy

import wradlib.verify


def flea(gauges, estimates):
    obs = gauges['precipitation'].values
    est = estimates['precipitation'].values
    obs = obs.flatten()
    est = est.flatten()
    mask = ~numpy.isnan(obs) & ~numpy.isnan(est)
    obs = obs[mask]
    est = est[mask]
    metrics = wradlib.verify.ErrorMetrics(obs, est)
    metrics.pprint()
    matplotlib.pyplot.scatter(obs,est)
    matplotlib.pyplot.show()
    return metrics.all()