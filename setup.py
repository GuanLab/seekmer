#!/usr/bin/env python3

import sys

import Cython.Build
import numpy
import setuptools
import setuptools.extension


_DIRECTIVES = {'language_level': '3'}


if 'CYTHON_TRACE_NOGIL' in sys.argv:
    _DIRECTIVES['linetrace'] = True

setuptools.setup(
    name='Seekmer',
    version='2020.0.0',
    description='A fast RNA-seq mapping tool',
    author='Hongjiu Zhang, Yifan Wang, Ryan Mills, Yuanfang Guan',
    author_email='zhanghj@umich.edu, yifwang@umich.edu, '
                 'remills@umich.edu, gyuanfan@umich.edu',
    url='http://github.com/guanlab/seekmer',
    license='GPLv3+',
    keywords='bioinformatics',
    python_requires='>=3.5.0',
    packages=setuptools.find_packages(exclude=['tests']),
    ext_modules=Cython.Build.cythonize(
        [setuptools.extension.Extension('*', ['seekmer/*.pyx'])],
        compiler_directives=_DIRECTIVES,
    ),
    include_dirs=[numpy.get_include()],
    entry_points={
        'console_scripts': [
            'seekmer = seekmer.__main__:main',
        ],
    },
    package_data={
        'seekmer': ['*.pxd', '*.pyx', 'test/data/*'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        ('License :: OSI Approved '
         ':: GNU General Public License v3 or later (GPLv3+)'),
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    setup_requires=['cython', 'numpy', 'pytest-runner'],
    tests_require=['pytest', 'pytest-datadir'],
    install_requires=['logbook', 'numpy', 'pandas', 'tables', 'scipy'],
)
