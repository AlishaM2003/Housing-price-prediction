# Housing Price Prediction

Predicting median house value for California districts (1990 U.S. Census data), and comparing a Random Forest built from scratch against scikit-learn's implementation.

## The problem

Given demographic and location data for a California census block group, predict the median house value  and understand what's actually happening inside a Random Forest by building one from the ground up, not just calling a library.

## What's in this repo

- **`california.ipynb`**  main project: data loading, cleaning, EDA, feature engineering, and modelling on the California Housing dataset
- **`randomforest.ipynb`** Random Forest built from scratch (custom `DecisionTreeRegressorScratch` class with recursive tree-building and split-finding logic), benchmarked against scikit-learn's `RandomForestRegressor`
- **`boston_dataset.ipynb`**  earlier coursework notebook (Boston Housing dataset). Kept for reference, not the primary project  this dataset has since been removed from scikit-learn due to an ethical concern with one of its original features, so `california.ipynb` is the featured project here.
- **`data/`**  dataset files

## Approach

1. **Data cleaning**  handled missing values and other data quality issues in the raw California Housing dataset
2. **Feature engineering** built ratio-based features (e.g. rooms per household) rather than relying on raw totals
3. **Modelling**  trained scikit-learn's `RandomForestRegressor` as a baseline
4. **Built from scratch**  implemented a Decision Tree Regressor and Random Forest entirely from scratch in NumPy (recursive splitting, MSE-based best-split search, no scikit-learn tree code used), then compared it directly against the library version

## Results

| Metric | Scikit-learn | From scratch |
|---|---|---|
| R² Score | **0.8923** | 0.8545 |
| MSE | 7.90 | 10.67 |
| Training Time | 0.69s | 65.68s |

The scratch implementation gets close to scikit-learn's accuracy, which confirms the underlying logic (splitting, tree-building) is correct. The big gap in training time (0.69s vs 65.68s) is expected and is actually the more interesting result  it shows exactly why production ML libraries are written in optimised/compiled code rather than pure Python: the algorithm is the same, but the implementation efficiency is not.


## What I learned

Building the tree from scratch forced me to actually understand what a Random Forest is doing at each split not just treat it as a black box. Comparing training times also gave me a concrete appreciation for why libraries like scikit-learn are engineered the way they are, rather than just being "the standard tool to reach for."

## Next Steps

 Compare against XGBoost or LightGBM to see whether gradient boosting beats Random Forest on this dataset
 Engineer a "distance to nearest major city" feature  likely a strong predictor not captured by raw latitude/longitude
 Try log-transforming the target variable, since house prices are right-skewed
 Test the from-scratch implementation on a second dataset to confirm it generalises, not just fits this one

## How to run it
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tabulate
```
Then open `california.ipynb` and `randomforest.ipynb` in Jupyter and run top to bottom.
