from setuptools import setup, find_packages

setup(
    name="house_price_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",
        "flask",
        "mlflow",
        "joblib",
        "pyyaml",
    ],
)