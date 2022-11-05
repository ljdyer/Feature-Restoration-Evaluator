from setuptools import setup
from setuptools import find_packages

REQUIREMENTS = [
    'pandas',
    'sklearn',
    'tqdm',
    'uniseg',
    'jiwer',
    'Jinja2',
]

setup(
    name='frmg',
    version='1.0',
    description='',
    author='Laurence Dyer',
    author_email='ljdyer@gmail.com',
    url='https://github.com/ljdyer/Feature-Restorer-Metric-Getter',
    package_dir={'': 'src'},
    packages=['src.feature_restorer_metric_getter'],
    install_requires=REQUIREMENTS
)
