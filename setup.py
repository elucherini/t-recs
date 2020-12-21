from setuptools import setup, find_packages


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="trecs",
    version="0.1.0b",
    url="https://github.com/elucherini/t-recs",
    license="GPL-3.0",
    author="Elena Lucherini",
    author_email="elucherini@cs.princeton.edu",
    description="Framework for simulating sociotechnical systems.",
    long_description=readme(),
    packages=find_packages(),
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.4.1",
        "networkx>=2.4",
        "tqdm>=4.46.0",
    ],
    zip_safe=False,
)
