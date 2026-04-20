import os

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = os.environ.get("SECRET_KEY", "boxing-hub-dev-secret-change-in-prod")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_DAYS = 30

DB_PATH    = os.environ.get("DB_PATH", os.path.join(_DIR, "data", "boxing_web.db"))
RF_PATH    = os.path.join(_DIR, "models", "random_forest.joblib")
SCALER_PATH = os.path.join(_DIR, "models", "rf_scaler.joblib")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
