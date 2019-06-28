# Counterfactual Fairness
Implementation and analysis of counterfactual fairness for my Bachelor's thesis. The implementation is based on the [code](https://github.com/mkusner/counterfactual-fairness) by the original authors of the [paper](https://arxiv.org/pdf/1703.06856.pdf) on counterfactual fairness.

The data set that is used is the [German Credit Data (GCD)](https://archive.ics.uci.edu/ml/datasets/statlog+%28german+credit+data%29) data set. The task associated with this data set is binary classification for credit risk (good or bad) of customers.

Some of the code files in the repository require `*.Rdata` files, which can be obtained by running the code and saving parts of it. The `gcd_preprocssing.R` file has to be run first to generate a `.Rdata` file of the pre-processed data set. From this point onwards, the rest of the required `.Rdata` files can be mostly obtained by just running the code and uncommenting the `save(x, file='*.Rdata')` lines. Lastly, it is important to set the proper working directory in your local development environment (e.g., in RStudio).
