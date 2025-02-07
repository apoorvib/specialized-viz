from setuptools import setup, find_packages

setup(
    name="specialized-viz",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
    'numpy>=1.20.0',
    'pandas>=1.3.0',
    'matplotlib>=3.4.0',
    'plotly>=5.3.0',
    'networkx>=2.6.0',
    'scikit-learn>=0.24.0',
    'scipy>=1.7.0',
    'statsmodels>=0.13.0',
    'pywt>=1.1.0',  # For wavelet analysis
    'fbprophet>=0.7.1',  # For time series forecasting
    'tensorflow>=2.6.0',  # For LSTM models
    'optuna>=2.10.0',  # For hyperparameter optimization
    'shap>=0.40.0',  # For model interpretability
    'community>=1.0.0b1',  # For community detection
    'python-louvain>=0.15',  # For Louvain community detection
    'nbeats-pytorch>=1.3.0',  # For N-BEATS forecasting
    'river>=0.7.0',  # For online learning
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.9.0',
            'black>=21.5b2',
            'isort>=5.8.0',
            'mypy',
            ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5.0',
            'nbsphinx>=0.8.0',
        ],
    },
    author="Apoorv Indrajit Belgundi",
    author_email="abelgundi@gmail.com",
    description="A specialized financial data visualization library focusing on time series, candlestick patterns, and network analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/apoorvib/specialized-viz",
        classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires='>=3.8'
)