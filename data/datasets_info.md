The datasets considered in this work are widely used tabular benchmark datasets. For each dataset, we provide the processed version in `data_splitted/` and the corresponding discretized version in `data_splitted_discretized/`.

The processed versions include preprocessing steps such as missing-value handling, feature transformations, one-hot encoding of categorical variables, conversion of binary string features to `0/1` values, and renaming of columns to clearer semantic labels. For further details on these processed dataset versions, please refer to:

https://huggingface.co/mstz/datasets

## Dataset sources

* adult - https://archive.ics.uci.edu/dataset/2/adult
* bank - https://archive.ics.uci.edu/ml/datasets/bank+Marketing
* breast - https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)
* compas - https://github.com/propublica/compas-analysis
* contr - https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice
* elect - https://www.openml.org/search?exact_name=electricity&type=data
* german - https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
* heart - https://archive.ics.uci.edu/ml/datasets/statlog+(heart)
* heloc - https://community.fico.com/s/explainable-machine-learning-challenge
* iris - https://archive.ics.uci.edu/ml/datasets/iris
* lrs - https://archive.ics.uci.edu/dataset/93/low+resolution+spectrometer
* magic - https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
* pol - https://www.openml.org/search?id=43983&type=data
* sonar - https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
* spam - https://archive.ics.uci.edu/ml/datasets/spambase
* steel - https://archive.ics.uci.edu/ml/datasets/steel+plates+faults
  We consider a simplified binary classification version of this problem: whether the input belongs to class `0` or not.
* stud - https://archive.ics.uci.edu/ml/datasets/student+performance
* wine - https://www.kaggle.com/datasets/ghassenkhaled/wine-quality-data
