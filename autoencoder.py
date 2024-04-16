import keras
from keras.losses import mean_squared_error

def autoencoder(n_input, n_bottleneck, n_layers):
    """This function makes an autoencoder with encoder, decoder, and complete model for training
    
    Args:

    Rules:
    1. Must be descending in size
    2. No layer may be larger than the input
    3. Let's have reasonableness about step sizes, weird -> (20, 19, 3)
    """

    inputs = keras.layers.Input(shape=(n_input, ))
    x = inputs

    #inputs of next layer are outputs of previous layer
    for layer_size in [n_input] + n_layers:
        x = keras.layers.Dense(layer_size)(x)

    #Bottleneck layer of autoencoder
    bottleneck = keras.layers.Dense(n_bottleneck)(x)


    #Decoder inputs
    dec_inputs = keras.layers.Dense(n_layers[-1], activation='relu')(bottleneck)
    y = dec_inputs

    for layer_size in n_layers[1::-1] + [n_input]:
        y = keras.layers.Dense(layer_size, activation='relu')(y)
    

    encoder_model = keras.models.Model(inputs=inputs, outputs=bottleneck)
    full_model = keras.models.Model(inputs=inputs, outputs=y)

    full_model.compile(loss=mean_squared_error, optimizer='adam')
    return encoder_model, full_model

if __name__ == '__main__':

    import read_wine_data
    import wine_classifier
    import pandas as pd

    df = read_wine_data.read('winequality-red.csv')

    #Defining features and labels
    labels = df['quality']
    features = df[[col for col in df.columns if col != 'quality']]


    #Scaling features
    for col in features:
        features[col] -= features[col].min()
        features[col] /= features[col].max()

    #Instantiating classifiers
    classifier = wine_classifier.WineClassifier('data', 'xgb')
    
    #Fitting classifier on two dimensionalities
    classifier.train(features=features, labels=labels, scaling='Min/Max')


    print(f'Classifier Metadata: {classifier.metadata}')


