import matplotlib.pyplot
import numpy

import wradlib.verify


def beaver(gauges, estimates):
    obs = gauges['precipitation'].values
    est = estimates['precipitation'].values
    obs = obs.flatten()
    est = est.flatten()
    mask = ~numpy.isnan(obs) & ~numpy.isnan(est)
    obs = obs[mask]
    est = est[mask]
    metrics = wradlib.verify.ErrorMetrics(obs, est)
    metrics.pprint()
    results = metrics.all()

    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(obs, est, alpha=0.5, edgecolors="k", linewidths=0.5)

    lims = [0, max(obs.max(), est.max()) * 1.1]
    ax.plot(lims, lims, "--", color="grey", linewidth=1, label="1:1")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed [mm]")
    ax.set_ylabel("Estimated [mm]")
    ax.set_title(
        f"n={len(obs)}  bias={results['meanerr']:.2f}  "
        f"RMSE={results['rmse']:.2f}  R={results['corr']:.2f}"
    )
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()

    return results, fig
