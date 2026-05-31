from train import model_training
from preprocess import preprocess
from predicting import prediction
import logging
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"../logs/pipeline_{timestamp}.log"

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"../logs/pipeline_{timestamp}.log"
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    filename=log_filename
    ) 

    logging.info("---------- Executing main script ----------")
    preprocess()
    model_training()
    prediction()


