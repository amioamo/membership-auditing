# Membership Auditing beyond a Single Reference Population

Reference implementation of a cross-population membership audit. The audit declares a class of non-member score laws, builds a lower confidence bound on the worst-case calibration threshold in that class, and returns an upper bound on the fraction of training records above that threshold.

For a Kolmogorov ball the threshold is an order statistic of the reference scores (`robust_order_stat_certificate`). A full member count uses `robust_order_stat_census`; a without-replacement member sample uses `robust_order_stat_hypergeom`. For a mixture class the threshold is the linear-program lower confidence bound `mixture_threshold_lcb`. The induced Kolmogorov radius of a mixture class is `induced_radius_lp`.

Scores are supplied by the caller. The package does not train a model or read a dataset.

Associated manuscript: *Membership Auditing beyond a Single Reference Population*.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. Import the package with `src` on `PYTHONPATH`.

## License

MIT License. See [LICENSE](LICENSE).
