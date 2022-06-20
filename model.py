import warnings
import gym
import constants
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

def get_model(env) -> Sequential:
    model = Sequential([
        
        InputLayer(input_shape = (constants.IMG_DIM_Y, constants.IMG_DIM_X, 1)),
        Conv2D(filters = 32, strides=1, kernel_size = 5, padding = "same", activation = "relu"), 
        MaxPooling2D(pool_size = 2, strides=2),
        Conv2D(filters = 16, strides=1, kernel_size = 3, padding = "valid", activation = "relu"), 
        MaxPooling2D(pool_size = 2, strides=2),
        Conv2D(filters = 8, strides=1, kernel_size = 3, padding = "valid", activation = "relu"), 
        MaxPooling2D(pool_size = 2, strides=2),
        Flatten(), 
        Dense(8, activation="relu"), 
        Dense(env.action_space.n, activation="linear") 
    ])

    opt = Adam(learning_rate = constants.LEARNING_RATE)
    loss = MeanSquaredError()
    model.compile(
        optimizer = opt, 
        loss = loss, 
        metrics = [loss]
    )

    # Check model weights if they exist.
    try:
        print("Weights loaded successfully.")
        model.load_weights(constants.WEIGHTS_PATH)
    except:
        pass

    return model

def save_model(model):
    print("Weights saved successfully.")
    model.save_weights(constants.WEIGHTS_PATH)