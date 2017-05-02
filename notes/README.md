|             | Decision Boundaries (Predict) | Learning | Parameters | Optimal | Variants | Efficient? |
| ----------- | :-------------------: | :-------:  | :----------: | :-------: | :-------: | :------: | 
| Perceptron  | Linear, Deterministic | Iterative, Online, Batch | Learning Rate | Yes, if linearly separable | Voted, Average | Relatively in -> Learning, Prediction |
| Logistic Regression | Linear, Probabilistic | Iterative, Online, Batch | Learning Rate, Lambda Regularization | Yes | Negative Log Likelihood | Similar to Perceptron |
| Naive Bayes | Linear for binary/count features | Maximum Likelihood Estimation | Smoothing | Yes (MLE) | Multinomial | Very -> Single scan of data |
| K Nearest Neighbor | Arbitrarily Complex, Deterministic | No Learning | K, Dist. Function | N/A Hueristic | | Very inefficient |
| Decision Tree | Axis-Parallel lines, Arbitrarily Complex | Greedy, Top-down Induction | Depth, Min # in Leaf, Gini or Entropy | No | | Yes |
| Neural Networks | Highly non-linear | Iterative, Batch -> Feed-Forward, Back-Propagation | Structure, Learning Rate, Initialization, Regularization, Dropout | No | Multi-Layer Perceptron | Not Efficient |
| Support Vector Machine | Linear, Non-linear(kernel) | Quadratic Program Solver | Soft-margin penalty, Kernel | No | | Linear -> Yes, Nonlinear -> less so| 
