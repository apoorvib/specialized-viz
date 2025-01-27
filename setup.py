from setuptools import setup, find_packages

setup(
    name="specialized-viz",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'networkx',
        'plotly',
        'scikit-learn',
        'ta-lib'
    ],
    author="Apoorv Indrajit Belgundi",
    author_email="abelgundi@gmail.com",
    description="A specialized data visualization library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/apoorvib/specialized-viz",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)