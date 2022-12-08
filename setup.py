# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os
from os import path, environ
from distutils.extension import Extension
from treeEnt.version import version as __version__
from shutil import copyfile
import platform, sys


# default args (that are modified per system specs below)
EXTRA_COMPILE_ARGS = []
DEFAULT_LIBRARY_DR = [f'{os.environ["CONDA_PREFIX"]}/include']

# setup
here = path.abspath(path.dirname(__file__))
system = platform.system()
py_version = str(sys.version_info.major) + str(sys.version_info.minor)

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# compile
kwargs = {'name':'treeEnt',
          'version':__version__,
          'description':'Entropy estimation for factorizable graphs',
          'long_description':long_description,
          'long_description_content_type':'text/markdown',
          'url':'https://github.com/eltrompetero/treeEnt',
          'author':'Edward D. Lee',
          'author_email':'edlee@csh.ac.at',
          'license':'MIT',
          'classifiers':['Development Status :: 5 - Production/Stable',
                         'Intended Audience :: Science/Research',
                         'Topic :: Scientific/Engineering :: Information Analysis',
                         'License :: OSI Approved :: MIT License',
                         'Programming Language :: Python :: 3 :: Only',
                        ],
          'python_requires':'>=3.10,<3.11',
          'keywords':'entropy estimation',
          'packages':find_packages(),
          'install_requires':['multiprocess>=0.70.7,<1',
                              'scipy',
                              'matplotlib',
                              'numpy',
                              'dill'],
          'include_package_data':True,  # see MANIFEST.in
          'py_modules':['treeEnt.measures',
                        'treeEnt.test_measures']}

setup(**kwargs)
