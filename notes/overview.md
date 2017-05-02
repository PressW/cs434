|             | Decision Boundaries (Predict) | Learning | Parameters | Optimal | Variants | Efficient? | Regularization |
| ----------- | :-------------------: | :-------:  | :----------: | :-------: | :-------: | :------: | :------: |
| Perceptron  | Linear, Deterministic | Iterative, Online, Batch | Learning Rate | Yes, if linearly separable | Voted, Average | Relatively in -> Learning, Prediction | Yes -> reduce weights to minimize overfitting |
| Logistic Regression | Linear, Probabilistic | Iterative, Online, Batch | Learning Rate, Lambda Regularization | Yes | Negative Log Likelihood | Similar to Perceptron | Yes -> using lambda parameter |
| Naive Bayes | Linear for binary/count features | Maximum Likelihood Estimation | Smoothing | Yes (MLE) | Very -> Single scan of data | |
