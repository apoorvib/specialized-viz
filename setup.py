from setuptools import setup, find_packages

setup(
    name="specialized-viz",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',  # Required for matplotlib visualization
        'plotly>=5.3.0',      # Required for plotly visualization
        'networkx>=2.6.0',
        'scikit-learn>=0.24.0',
        'yfinance'  # For example data
    ],# Removed ta-lib
    extras_require={
        'dev': [
            'pytest',
            'black',
            'flake8'
        ]
    }, 
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