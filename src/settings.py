from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).parent.parent/'.env'
load_dotenv(dotenv_path=env_path)
 
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")
DEFAULT_DATASETS_PATH = os.getenv("DEFAULT_DATASETS_PATH")


if not os.path.isdir(DEFAULT_DATASETS_PATH):
  os.makedirs(DEFAULT_DATASETS_PATH)
