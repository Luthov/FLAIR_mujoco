from setuptools import setup, find_packages

setup(
    name='speech_to_text',
    version='0.0.0',
    packages=find_packages(where='src'),  # Add any other packages here
    package_dir={'': 'src'},
)