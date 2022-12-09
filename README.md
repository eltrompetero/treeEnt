# TreeEnt: Estimating the entropy of factorizable graphs

TreeEnt is a Python project for estimating the entropy of probabilistic graph models by leveraging bridges between communities and a good estimator of entropy on sparse samples. We show that using the structural constraint provides a good heuristic for identifying these bridges and partitioning the graph into conditionally independent copmonents. We combine this heuristic with the Nemenman-Shafee-Bialek estimator for entropy to obtain a faster and more accurate estimator and demonstrate its application to the pairwise maximum entropy, or Ising, model. The calculation can be extended to other models with a straightforward modification of the code when the graph structure, such as that given by interactions in an Ising model, are known and a model sampler is given. We demonstrate an application of this algorithm to estimate the entropy of judicial voting models of moderate size ($N=47$ and $N=23$), where entropy estimates without leveraging the graph structure are poor.

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
