
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
from distutils.command.build_py import build_py as _build_py
import sys, os.path
import re

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
import subprocess
import platform


setup(name='hrl4in',
    version='0.0.1',
    author='Stanford University',
    zip_safe=False)
