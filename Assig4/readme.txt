## Neural Network Model Evaluation Results

This readme file provides the results of evaluating the neural network models on the test set. The models were trained using different activation functions, dropout rates, and lambda values. The following table summarizes the results:

| Activation Function | Dropout Rate | Lambda | Test Loss | Test Accuracy |
|---------------------|--------------|--------|-----------|---------------|
| relu                | 0.3          | 0.001  | 0.5006    | 0.7676        |
| sigmoid             | 0.3          | 0.001  | 0.5718    | 0.7266        |
| tanh                | 0.5          | 0.001  | 0.5416    | 0.7349        |

Please note that the best model parameters refer to the parameters that achieved the highest Test accuracy during training.

1.The choice of activation function in the hidden layer has a notable effect on the classification accuracy. ReLU (Rectified Linear Unit) performs the best among the three activation functions, achieving the highest test accuracy of 0.7676. ReLU is known for its ability to mitigate the vanishing gradient problem and promote faster convergence during training.

2.Sigmoid activation function, despite being a popular choice for binary classification, yields slightly lower accuracy of 0.7266. Sigmoid's output is bounded between 0 and 1, which can result in vanishing gradients during backpropagation. This limitation may hinder the model's ability to capture complex patterns in the data.

3.Tanh activation function performs better than sigmoid, with a test accuracy of 0.7349. Tanh is a scaled and shifted version of the sigmoid function, ranging from -1 to 1. It addresses the vanishing gradient problem to some extent, but still suffers from the saturation issue at the extremes, affecting its ability to learn complex patterns.

4.The addition of L2-norm regularization, controlled by the lambda value, helps prevent overfitting by imposing a penalty on the model's weights. In our experiments, a small lambda value of 0.001 is used. This regularization has a marginal positive effect on the results, as it slightly improves the test accuracy for all activation functions.

5.Dropout regularization, with a dropout rate of 0.3 or 0.5, randomly deactivates a fraction of neurons during training, forcing the model to learn more robust representations. The dropout technique is effective in preventing overfitting, and in our case, it improves the test accuracy for all activation functions. However, it has a more significant positive impact on the sigmoid activation function, which tends to overfit more easily.

6.In summary, the choice of activation function significantly affects the model's performance, with ReLU performing the best among the three options. L2-norm regularization and dropout both help improve generalization and prevent overfitting, with dropout exhibiting a more substantial impact on the sigmoid activation function. The results highlight the importance of carefully selecting these parameters to achieve optimal performance in neural network models.

source /media/data/msci641/gsahu/miniconda3/bin/activate /media/data/msci641/gsahu/miniconda3/envs/dev/