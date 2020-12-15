from pathlib import Path
from setuptools import setup, find_packages

with open("README.md") as fp:
    long_description = fp.read()

with Path("pewlib", "__init__.py").open() as fp:
    for line in fp:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')

setup(
    name="pewlib",
    version=version,
    description="Import, processing and export library for LA-ICP-MS data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="T. Lockwood",
    author_email="thomas.lockwood@uts.edu.au",
    url="https://github.com/djdt/pewlib",
    project_urls={
        "Documentation": "https://djdt.github.io/pewlib",
        "Source": "https://gtihub.com/djdt/pewlib",
    },
    packages=find_packages(include=["pewlib", "pewlib.*"]),
    install_requires=["numpy"],
    tests_require=["pytest"],
)
