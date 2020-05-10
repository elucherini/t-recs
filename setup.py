from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='rec',
        version='0.1',
        url='https://github.com/elucherini/algo-segregation',
        license='GPL-3.0',
        author='Elena Lucherini',
        author_email='elucherini@cs.princeton.edu',
        description='Framework for simulating sociotechnical systems.',
        long_description=readme(),
        packages=find_packages(),
        install_requires=[
            'numpy>=1.17.0',
            'scipy>=1.4.1',
            'pandas>=0.25.0',
            'networkx>=2.4',
            'tqdm>=4.46.0',
        ],
        zip_safe=False)
