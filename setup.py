from setuptools import setup, find_packages

from pew import __version__

setup(
    name="pew",
    description="Import / export library for LA-ICP-MS data.",
    author="djdt",
    version=__version__,
    packages=find_packages(include=["pew", "pew.*"]),
    install_requires=["numpy"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
