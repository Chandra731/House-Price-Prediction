import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_models(X_train, y_train, config):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=config["training"]["random_state"]),
        "Random Forest": RandomForestRegressor(random_state=config["training"]["random_state"]),
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", random_state=config["training"]["random_state"])
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models