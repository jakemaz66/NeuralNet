import tensortango.train as train, tensortango.layer as layer, tensortango.mlp as mlp

if __name__ == '__main__':

    #Readingin data
    import read_wine_data
    df = read_wine_data.read('winequality-red.csv')

    labels = df['quality']
    features = df[[col for col in df.columns if col != 'quality']]

    #Scaling features
    for col in features:
        features[col] -= features[col].min()
        features[col] /= features[col].max()
    
    #Defining mulit-layer perceptron
    neural_net = mlp.MLP([layer.Relu(11,8),
                          layer.Relu(8,6),
                          layer.Relu(6,4),
                          layer.Tanh(4,2),
                          layer.Relu(2,4),
                          layer.Relu(4,6),
                          layer.Relu(6,8),
                          layer.Relu(8,11)])
    
    train.train(neural_net, features.values, features.values)
    print(neural_net.forward(features.values))