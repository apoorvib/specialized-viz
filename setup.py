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
        'yfinance',
        'statsmodels>=0.13.0',  # For VAR models
        'fbprophet>=0.7.1',     # For Prophet models
        'tensorflow>=2.8.0',    # For LSTM
        'nbeats-pytorch>=1.3.0',  # For N-BEATS
        'optuna>=2.10.0',      # For hyperparameter optimization
        'river>=0.8.0',        # For online learning
        'scipy>=1.7.0',
        'shap>=0.40.0'         # For feature importance
    ],
    extras_require={
        'dev': [
            'pytest',
            'black',
            'flake8',
            'mypy',
            'isort'
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
    python_requires='>=3.7'
)