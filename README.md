# Closely estimating the entropy of factorizable graph models

We introduce an algorithm for estimating the entropy of pairwise, probabilistic graph models by leveraging
bridges between social communities and an accurate entropy estimator on sparse samples. We propose using a
measure of investment from the sociological literature, Burt's structural constraint, as a heuristic for
identifying bridges that partition a graph into conditionally independent components. We combine this
heuristic with the Nemenman-Shafee-Bialek entropy estimator to obtain a faster and more accurate estimator. We
demonstrate it on the pairwise maximum entropy, or Ising, models of judicial voting, to improve na\"{i}ve
entropy estimates. We use our algorithm to estimate the partition function closely, which we then apply to the
problem of model selection, where estimating the likelihood is difficult. This serves as an improvement over
existing methods that rely on point correlation functions to test fit can be extended to other graph models
with a straightforward modification of the open-source implementation.

For additional background, see the arXiv preprint: [arxiv.org]

# Installation
```bash
python setup.py bdist_wheel
pip install dist/*.whl
```

# Troubleshooting
If there is a conflict in Python version, then one can force pip to install TreeEnt with the current version
of Python using
```bash
pip install --no-deps --ignore-requires-python dist/*.whl
```
