from setuptools import setup


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="trecs",
    version="0.2.1",
    description="Framework for simulating sociotechnical systems.",
    url="https://github.com/elucherini/t-recs",
    license="MIT",
    author="Eli Lucherini",
    author_email="elucherini@cs.princeton.edu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=readme(),
    packages=[
        "trecs",
        "trecs.models",
        "trecs.metrics",
        "trecs.components",
        "trecs.base",
        "trecs.random",
    ],
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.4.1",
        "networkx>=2.4",
        "tqdm>=4.46.0",
        "lenskit>=0.11.1",
        "pandas==1.0.5",
    ],
    zip_safe=False,
)
