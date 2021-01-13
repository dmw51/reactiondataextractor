from setuptools import setup, find_packages

setup(
    name='reactiondataextractor',
    version='0.0.1',
    author='Damian Wilary',
    author_email='dmw51@cam.ac.uk',
    packages=find_packages(),
    install_requires=[
        'chemdataextractor',
        'pyosra',
        'tesseract>=4.1',
        'tesserocr>=2.5'

    ]
)

