from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'DeepKIN Ag-AI'
LONG_DESCRIPTION = 'Kinyarwanda Deep Learning Models and Tools for IVR/RAG-based Chatbots'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="deepkin",
    version=VERSION,
    author="Antoine Nzeyimana",
    author_email="<nzeyi@kinlp.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
    ],
    keywords=['python', 'deepkin', 'kinyarwanda', 'nlp', 'deep learning', 'agriculture', 'ai'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Research and Development",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux :: Linux OS",
    ]
)
