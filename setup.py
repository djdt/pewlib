from setuptools import setup, find_packages

from pew import __version__

setup(
    name="pewlib",
    description="Import, export and processing library for LA-ICP-MS data.",
    url="https://github.com/djdt/pew",
    author="T. Lockwood",
    author_email="thomas.lockwood@uts.edu.au",
    version=__version__,
    packages=find_packages(include=["pew", "pew.*"]),
    install_requires=["numpy"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    project_urls={
        "Documentation": "https://djdt.github.io/pew",
        "Source": "https://gtihub.com/djdt/pew",
    },
)
