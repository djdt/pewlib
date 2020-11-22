from setuptools import setup, find_packages

with open("README.md") as fp:
    long_description = fp.read()

setup(
    name="pewlib",
    description="Import, processing and export library for LA-ICP-MS data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djdt/pewlib",
    author="T. Lockwood",
    author_email="thomas.lockwood@uts.edu.au",
    version="0.6.2",
    packages=find_packages(include=["pew", "pewlib.*"]),
    install_requires=["numpy"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    project_urls={
        "Documentation": "https://djdt.github.io/pewlib",
        "Source": "https://gtihub.com/djdt/pewlib",
    },
)
