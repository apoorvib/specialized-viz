from setuptools import setup, find_packages

setup(
    name="patternforge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.1.0',
        'matplotlib>=3.8.0',
        'plotly>=5.18.0',
        'networkx>=3.2.0',
        'scikit-learn>=1.3.0',
        'yfinance>=0.2.32',
        'statsmodels>=0.14.0',
        'tensorflow>=2.15.0',
        'optuna>=3.4.0',
        'river>=0.20.0',
        'scipy>=1.11.0',
        'shap>=0.43.0',
        'pmdarima>=2.0.3',  # Added instead of fbprophet which is deprecated
        'PyWavelets>=1.5.0',  # Changed from pywt to PyWavelets
        'nbeats-pytorch>=1.3.0',
        'python-louvain>=0.16',
        'community>=1.0.0b1',
        'nbformat>=5.9.0',
        'ipywidgets>=8.1.0',
        'ipykernel>=6.27.0',
        'jupyter>=1.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'flake8>=6.1.0',
            'black>=23.11.0',
            'isort>=5.12.0',
            'mypy>=1.7.0',  # Added version
        ],
        'docs': [
            'sphinx>=7.2.0',
            'sphinx-rtd-theme>=1.3.0',
            'nbsphinx>=0.9.3',
        ],
    },
    author="Apoorv Indrajit Belgundi",
    author_email="abelgundi@gmail.com",
    description="A specialized financial data visualization library focusing on time series, candlestick patterns, and network analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/apoorvib/patternforge",  # Updated to new name
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",  # Added newer Python versions
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires='>=3.8'
)