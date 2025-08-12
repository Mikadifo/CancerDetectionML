# Report

## Algorithms

This program lets the user choose a Random Forest Algorithm or a Gradient Boosting Algorithm to predict if a tumor id bening or malignant.

Both Algorithms work around decision trees, and are ensembles, meaning that we have multiple models working at the same time with different weights and parameters to reduce error.

Random Forest helps us reduce overfitting and variance, since it's robust in terms of data noise.

Gradient Boosting helps us improve our prediction based iteratively on previous predictions. However it is not as strong as random forest in terms of data noise.

In general, the features of both algorithms make them perfect for this binary classification problem.

## Description of your training procedure

For each algorithm we do these steps:

1. Read the csv file into a dataFrame
2. Transform the dataFrame object to an RDD where the 'diagnosis' column will be 0.0 if it's benign (B) and 1.0 if it's malignant (M)
3. We split the data into training data (70%) and testing data (30%)
4. Train models with a depth of 5 and 50 trees for randomForest and 50 iterations for gradientBoosting
5. Collect predictions and calculate the testing error

## Testing results

### Random Forest

**Test Error:** 0.07065217391304347

**Correct Predictions:** 171

**Incorrect Predictions:** 13

**Confusion Matrix:** [[111, 4], [9, 60]]

### Gradient Boosting

**Test Error:** 0.06321839080459771

**Correct Predictions:** 163

**Incorrect Predictions:** 11

**Confusion Matrix:** [[104, 5], [6, 59]]

## Evaluation

### Random Forest

**F1 Score:** 0.9253731343283582

**Precision:** 0.96875

**Recall:** 0.8857142857142857

**Accuracy (% correct predictions):** 94.56521739130434 %

### Gradient Boosting

**F1 Score:** 0.9147286821705427

**Precision:** 0.921875

**Recall:** 0.9076923076923077

**Accuracy (% correct predictions):** 93.67816091954023 %

## Comparison and analysis of the two algorithms

Based on the evaluation above, both algorithms performed similarly, with a high accuracy.

Random Forest had a slightly higher accuracy and precision than Gradient Boosting. Howerver, RF had less recall than GB. Similarly, Random Forest did a little bit better in the F1 Score than Gradient Boosting.

In conclusion, Random Forest performed better in most of the scores, but it wasn't a big difference from Gradient Boosting, so we could use either one for this problem.

## Limitations of your approach

- 569 rows might not be enough to correctly measure and train the machine learning algorithms
- Metrics and training might not be the best for current properties like depth, trees, iterations
- Not clear which algorithm will perfoms better in case we have a huge data set

## Suggestions for future improvements

- Add cross-validation to reduce variance
- Try different machine learning algorithms and more intense testing for each one
- Test with a larger dataset
- Improve tunning for each algorithm
