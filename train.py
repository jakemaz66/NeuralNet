import tensor, loss, mlp, optimizer, data_iterator

def train(nn: mlp.MLP,
          features:tensor.Tensor,
          labels:tensor.Tensor,
          eopchs: int = 5000,
          iterator=data_iterator.BatchIterator(),
          loss_fn=loss.MSE(),
          optimizer_obj=optimizer.SGD,
          learning_rate: float=0.05):
    """Train a neural network (MLP)

    Args:
        nn (mlp.MLP): Neural Network
        features (tensor.Tensor): Tensor of Features
        labels (tensor.Tensor): Tensor of Features
        eopchs (int, optional): Number of rounds of forward+backward training. Defaults to 5000.
        iterator (_type_, optional): Batch Iterator . Defaults to data_iterator.BatchIterator.
        loss_fn (_type_, optional): Loss function. Defaults to loss.MSE.
        optimizer_obj (_type_, optional): Mechanism to update learning rate. Defaults to optimizer.SGD.
        learning_rate (float, optional): . Amount of error to include in each backprop step to 0.05.
    """

    optm = optimizer_obj(nn, learning_rate)

    for epoch in range(eopchs):
        epoch_loss = 0.0

        for batch in iterator(features, labels):
            features, labels = batch

            predictions = nn.forward(features) #First result is features

            epoch_loss += loss_fn.loss(predictions, labels) #Labels are batch 1

            grad = loss_fn.gradient(predictions, labels)

            nn.backward(grad)

            optm.step()

            nn.zero_parameters()
        
        print(f'Epoch{eopchs} has loss {epoch_loss}')


if __name__ == '__main__':
    #Exclusive Or NN

    import numpy as np
    import layer

    #XOR because linear functions cannot represent
    features = np.array([[0,0],
                        [0,1],
                        [1,0],
                        [1,1]])
    
    labels = np.array([[1,0], #False is 1 true is 0
                       [0,1],
                       [0,1],
                       [1,0]])
    
    neural_net = mlp.MLP([layer.Tanh(2,2),
                          layer.Tanh(2,2)])
    
    train(neural_net, features, labels)
    print(neural_net.forward(features))


    
