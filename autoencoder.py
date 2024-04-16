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

    df = read_wine_data.read('winequality-red.csv')

    #Defining features and labels
    labels = df['quality']
    features = df[[col for col in df.columns if col != 'quality']]

    #Store scaling factors in metadata
    enc, full = autoencoder(n_input=11, n_bottleneck=2, n_layers=[8,6,4])
    enc2, full2 = autoencoder(n_input=11, n_bottleneck=4, n_layers=[8,6])

    #Scaling features
    for col in features:
        features[col] -= features[col].min()
        features[col] /= features[col].max()
    
    #An autoencoder is defined as output data = input data, features are labels because of autoencoder architecture
    full.fit(features.values, features.values, epochs=50, batch_size=32, shuffle=True)
    #Getting my reduced dimensionality feature set
    #ORIGINAL DIMENSIONALITY: (1599, 11)
    #NEW DIMENSIONALITY: (1599, 2)
    features_reduced = enc.predict(features)

    #NEW DIMENSIONALITY: (1599, 4)
    features_reduced2 = enc2.predict(features)

    #Instantiating classifiers
    classifier = wine_classifier.WineClassifier('data', 'xgb')
    classifier_reduced = wine_classifier.WineClassifier('data', 'xgb')
    classifier_reduced2 = wine_classifier.WineClassifier('data', 'xgb')
    
    #Fitting classifier on two dimensionalities
    classifier.train(features=features, labels=labels)
    classifier_reduced.train(features=features_reduced, labels=labels)
    classifier_reduced2.train(features=features_reduced2, labels=labels)

    print(f'Classifier Metadata: {classifier.metadata}')
    print(f'Classifier Reduced Dimensionality 2 Metadata: {classifier_reduced.metadata}')
    print(f'Classifier Reduced Dimensionality 4 Metadata: {classifier_reduced2.metadata}')


