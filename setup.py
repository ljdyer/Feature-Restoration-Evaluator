from distutils.core import setup

REQUIREMENTS = [
    'pandas',
    'sklearn',
    'tqdm',
    'uniseg',
    'jiwer',
    'Jinja2',
]

setup(
    name='Feature Restorer Metric Getter',
    version='1.0',
    description='',
    author='Laurence Dyer',
    author_email='ljdyer@gmail.com',
    url='https://github.com/ljdyer/Feature-Restorer-Metric-Getter',
    packages=['frmg', 'frmg.feature_restorer_metric_getter'],
    install_requires=REQUIREMENTS
)
