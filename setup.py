import numpy as np
import os.path as osp
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


ext_modules = [
    Extension(
        'torchreid.metrics.rank_cylib.rank_cy',
        ['torchreid/metrics/rank_cylib/rank_cy.pyx'],
        include_dirs=[numpy_include()],
    )
]


def get_requirements(filename='requirements.txt'):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires

setup(
    description='A library for deep learning person re-ID in PyTorch',
    license='MIT',
    long_description=readme(),
    packages=find_packages(),
    install_requires=get_requirements(),
    extras_require={"labels": get_requirements("requirements_labels.txt")},
    keywords=['Person Re-Identification', 'Deep Learning', 'Computer Vision'],
    ext_modules=cythonize(ext_modules)
)
