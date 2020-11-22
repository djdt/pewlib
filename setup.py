from setuptools import setup, find_packages

setup(
    name="pewlib",
    description="Import, processing and export library for LA-ICP-MS data.",
    url="https://github.com/djdt/pewlib",
    author="T. Lockwood",
    author_email="thomas.lockwood@uts.edu.au",
    version="0.6.0",
    packages=find_packages(include=["pew", "pew.*"]),
    install_requires=["numpy"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    project_urls={
        "Documentation": "https://djdt.github.io/pewlib",
        "Source": "https://gtihub.com/djdt/pewlib",
    },
)
