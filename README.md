# House Price Prediction MLOps Project

## Overview
This project aims to create a machine learning model to predict house prices. The project follows MLOps practices to ensure reproducibility, scalability, and maintainability.

## Project Structure
```
housing-price-prediction-mlops/
│── .github/
│   └── workflows/
│       └── mlops-pipeline.yml    # GitHub Actions for automation
│
├── artifacts/                    # Stores trained models & preprocessing objects
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│
├── config/                       # Configuration files
│   ├── config.yaml               # Model & preprocessing hyperparameters
│   ├── logging.yaml              # Logging configurations
│
├── data/                         # Raw & processed datasets
│   ├── raw/
│   │   ├── housing.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── test.csv
│
├── notebooks/                    # Jupyter notebooks for EDA & testing
│   ├── EDA.ipynb                 # Exploratory Data Analysis
│   ├── Feature_Engineering.ipynb  # Feature transformation experiments
│   ├── Model_Training.ipynb       # Model training experiments
│
├── src/                          # Source code for modularized ML pipeline
│   ├── __init__.py
│   ├── data_preprocessing.py     # Handling missing values, encoding, scaling
│   ├── feature_engineering.py    # Feature selection, transformation
│   ├── model_training.py         # Training models (Linear, DecisionTree, XGBoost)
│   ├── hyperparameter_tuning.py  # RandomizedSearchCV for hyperparameter tuning
│   ├── evaluation.py             # Model evaluation & residual analysis
│   ├── save_artifacts.py         # Save model & preprocessing artifacts
│
├── scripts/                      # Automation scripts
│   ├── train.py                  # Executes full training pipeline
│   ├── predict.py                # Uses trained model for predictions
│
├── app/                          # Deployment (Streamlit or Flask API)
│   ├── app.py                    # Flask/FastAPI endpoint for inference
│   ├── templates/
│       └── index.html            # HTML template for user input
│
├── tests/                        # Unit tests for MLOps pipeline
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│
├── Dockerfile                    # Dockerfile for containerization
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment setup
├── setup.py                      # Package installation
├── .gitignore                    # Ignore unnecessary files
├── README.md                     # Project documentation
```

## Getting Started
### Prerequisites
- Docker
- Docker Compose
- Conda (optional)

### Installation
1. Clone the repository
   ```
   git clone <repository_url>
   cd housing-price-prediction-mlops
   ```

2. Set up the environment using Conda or Pip

#### Using Conda:
```
conda env create -f environment.yml
conda activate housing-price-prediction-mlops
```

#### Using Pip:
```
pip install -r requirements.txt
```

3. Prepare the Data
Ensure that your raw data file (`housing.csv`) is placed in the `data/raw/` directory.

4. Train the Model
Run the training script to preprocess data, train the model, and save the artifacts:
```
python scripts/train.py
```
This script will:
- Load and preprocess the data.
- Train multiple models and evaluate them.
- Perform hyperparameter tuning on the best model.
- Save the trained model and preprocessing artifacts.
- Save the processed train and test datasets.

5. Run the Flask App
Start the Flask app to serve the model for predictions:
```
python app/app.py
```
The Flask app will be available at `http://localhost:5000`.

6. Make Predictions
Use the web form at `http://localhost:5000` to input data and get predictions directly on the web page.

7. Run Tests
Run the unit tests to ensure everything is working correctly:
```
pytest
```

8. Dockerization (Optional)
You can also run the project inside a Docker container.

#### Build the Docker Image:
```
docker build -t housing-price-prediction-mlops .
```

#### Run the Docker Container:
```
docker run -p 5000:5000 housing-price-prediction-mlops
```

9. CI/CD with GitHub Actions
If you have set up GitHub Actions, the pipeline will automatically run on push or pull request to the `main` branch. The workflow file is located at `.github/workflows/mlops-pipeline.yml`.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License
This project is licensed under the MIT License.