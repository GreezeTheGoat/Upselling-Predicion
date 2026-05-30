from train import model_training
from preprocess import preprocess
from predicting import prediction

if __name__ == "__main__":
    print("---------- Executing main script ----------")
    preprocess()
    model_training()
    prediction()


