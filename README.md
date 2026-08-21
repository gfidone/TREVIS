<p align="center">
  <img src="docs/trevis_logo.png" alt="TREVIS logo" width="350"/>
</p>

TREVIS (Tree REpresentations from Variational Inference in latent Space) is a generative approach to Decision Tree (DT) learning based on the exploration of the latent space of a Tree Transformer Variational Auto-Encoder (TTVAE), allowing for optimization w.r.t. complex objectives.

# Usage

To run TREVIS, follow these steps:

1. **Download this repository on your local machine**.
2. **Fit random train DTS**. Edit the `config/random_trees_config.json` file with your preferred configuration and run:

   ```bash
    python3 scripts/random_trees.py 
    ```
4. **Tokenize random DTs**. Edit the `config/tokenize_config.json` file with your preferred configuration and run:

   ```bash
    python3 scripts/tokenize.py 
    ```
5. **Train the TTVAE**. Edit the `config/train_ttvae_config.json` file with your preferred configuration and run:

    ```bash
    python3 scripts/train_ttvae.py 
    ```

7. **Optimize via gradient ascent**. Edit the `config/gradient_config.json` file with your preferred configuration and run:

   ```bash
    python3 scripts/gradient.py 
    ```

Alternatively, set all configuration files in `config` and run:

  
  ```bash
  bash trevis.sh
  ```

to automatically run steps 2-5. Best found DTs are stored in `experiments/best_dts.json`

# Experiments

## Competitors

We compare against the following competitors. Please refer to the corresponding repositories for implementation details and requirements:

- `gosdt_lb`: [GOSDT with guesses](https://github.com/ubc-systopia/gosdt-guesses)
- `DL8.5`: [PyDL8.5](https://github.com/aia-uclouvain/pydl8.5)
- `DL8.5-lbguess`: [PyDL8.5 with guessed tighter lower bounds](https://github.com/ubc-systopia/pydl8.5-lbguess)
- `FlowOCT`: [ODTLearn FlowOCT implementation](https://github.com/D3M-Research-Group/odtlearn/blob/main/odtlearn/flow_oct.py)
- `CART`: [Scikit-learn DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

You can run the experiments for the competitors using the scripts in `competitors` folder. Also in this case, modify the config appropriately when in need of using models with respect to original or discretized features.

# Data

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


# License
TREVIS is distributed under the GNU General Public License. Refer to LICENSE.txt for details.
