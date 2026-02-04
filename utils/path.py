
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT,"logs")
# SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT,"saved_models")


# dataset_generator/gen_script.py
# RAW_DATA_DIR = os.path.join(PROJECT_ROOT,"_dataset_gen/processed")
# PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT,"_dataset_gen/dataset")
# IMAGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT,"_dataset_gen/dataset/images")
# LOG_DIR = os.path.join(PROJECT_ROOT,"data/logs/")

# preprocessing/dataset.py
IMAGE_DIR = os.path.join(PROJECT_ROOT,"_dataset_gen/dataset/images") 
TRAIN_CSV = os.path.join(PROJECT_ROOT,"_dataset_gen/dataset/Train.csv") 

# models/fg_mfn.py
MODEL_CONFIG = os.path.join(PROJECT_ROOT,"models/configs/model_config.json") 

# preprocessing/dataset_preprocessing.py
    # TRAIN_CSV = "data/processed/train.csv"
PROCESSED_IMAGE_DIR = os.path.join(PROJECT_ROOT,"_dataset_gen/dataset/images")
VAL_CSV = os.path.join(PROJECT_ROOT,"_dataset_gen/dataset//Val.csv")
TEST_CSV = os.path.join(PROJECT_ROOT,"_dataset_gen/dataset/Test.csv")

# training/train.py
    # TRAIN_CSV = "data/processed/train.csv"
    # VAL_CSV = "data/processed/val.csv"
#    MODEL_CONFIG = "models/configs/model_config.json"
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT,"saved_models")

# training/logger.py
    # LOG_DIR = "data/logs/"

# training/evaluate.py
    # TEST_CSV = "data/processed/test.csv"
    # MODEL_CONFIG = "models/configs/model_config.json"
    # LOG_DIR = "data/logs/"
SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT,"saved_models/model_best.pt")

# server/predict.py
    # SAVED_MODEL_PATH = "models/saved_models/model_final.pt"
    # MODEL_CONFIG = "models/configs/model_config.json"
IMAGE_UPLOAD_DIR = os.path.join(PROJECT_ROOT,"_dataset_gen/app_temp_images")


# server/app.py
    # SAVED_MODEL_PATH = "models/saved_models/model_final.pt"
    # UPLOAD_FOLDER = IMAGE_UPLOAD_DIR
    # LOG_DIR = "data/logs/"


print(f"Project Root Directory: {SAVED_MODEL_PATH} ")
print(f"Log Directory: {LOG_DIR} ")
print(f"Saved Model Directory: {SAVED_MODEL_DIR} ")
print(f"Image Upload Directory: {IMAGE_UPLOAD_DIR} ")
print(f"Model Config Path: {MODEL_CONFIG} ")
print(f"Train CSV Path: {TRAIN_CSV} ")
print(f"Processed Image Directory: {PROCESSED_IMAGE_DIR} ")
print(f"Saved Model Path: {SAVED_MODEL_PATH} ")
# print(f"Raw Data Directory: {RAW_DATA_DIR} ")
# print(f"Processed Data Directory: {PROCESSED_DATA_DIR} ")
# print(f"Image Output Directory: {IMAGE_OUTPUT_DIR} ")
print(f"Validation CSV Path: {VAL_CSV} ")
print(f"Test CSV Path: {TEST_CSV} ")
