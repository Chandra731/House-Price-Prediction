import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

def hyperparameter_tuning(X_train, y_train, random_state):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }
        
        model = xgb.XGBRegressor(objective="reg:squarederror", **params, random_state=random_state)
        
        X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
        
        model.fit(X_train_split, y_train_split)
        y_pred = model.predict(X_valid_split)
        rmse = mean_squared_error(y_valid_split, y_pred, squared=False)
        
        return rmse

    # Set a seed for reproducibility
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
    
    # Reduce the number of trials to 10 for debugging
    study.optimize(objective, n_trials=10)

    return study.best_params_