# Scale-Invariant Infinite Hierarchical Topic Model (ihLDA)

## Paper
<a href="https://shusei-e.github.io/" target="_blank">Shusei Eshima</a> and <a href="http://chasen.org/~daiti-m/index.html" target="_blank">Daichi Mochihashi</a>. 2023. Scale-Invariant Infinite Hierarchical Topic Model. In *Findings of the Association for Computational Linguistics: ACL 2023*.


## Requirements
Python 3.9.6
```
Cython==0.29.28
gensim==4.2.0
matplotlib==3.5.1
nltk==3.7
numpy==1.23.4
pandas==1.4.2
scikit-learn==1.0.2
```

## Usage

### Data Preparation
```bash
$ python preprocessing.py
```
The `input/sample_raw` folder contains ten sample documents for testing purposes (note that this is not enough data to obtain any meaningful results).

### Fitting the Model
```bash
$ python setup.py build_ext --inplace
$ python main.py --output_path ./output/

# or if the default settings are fine, just run
$ python run.py
```

The output folder contains the following files:
- `fig_tssb/`: the structure of the root tree.
- `model/`: the output is saved every 1000 iterations.
- `filenames.csv`: the list of file names and `doc_id`s.
- `info.csv`: the number of topics.
- `parameters.csv`: the hyperparameters for each iteration.
- `perplexity.csv`: the perplexity.
- `TopWords_prob.csv`: the topic-word distribution of top words.
- `model_temp.pkl`: the temporary model object. This allows us to resume the iteration, but the random seed will be reset if you resume the iteration.
- `txtdata.pkl`: the data object.
- `settings.txt`: the settings of the model.

### Evaluation
`evaluate.ipynb` calls the evaluation function in `evaluate_helper.py`.
