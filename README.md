# HS2914-auto-correction-project

The code and sentence examples used in the BERT-NCM part of our project on the topic "String edit distance and its applications" for HS2914. 

# BERT-NCM

Automatic sentence correction with the BERT and the Noisy Channel Model.

## Instructions

### Requirements

Simply run the following command to install the required Python libraries through pip:

```
pip3 install -r requirements.txt
```

In case your environment cannot install PyEnchant and PyTorch through pip properly, check out the following links:

- [Installation of PyEnchant](https://pyenchant.github.io/pyenchant/install.html)
- [Installation of PyTorch](https://pytorch.org/get-started/locally/)

### Correct one sentence from the command line

Use the following command: 

```python llm_correction.py --input "Replace with the sentence to be corrected."```

### Correct sentences in a csv file

Use the following command:

```python llm_correction.py --dataset_dir "./data.csv" --output_dir "./output.csv"```

### Debugging

By default, the warning/error messages are ignored. 

If an output is expected but not displayed, try add the ```--debug``` argument to display the warning/error messages.

```python llm_correction.py [...other arguments...] --debug```

## Hyperparameters

The most important hyperparameters of this approach are the ```alpha``` and ```gamma``` in the ```NoisyChannelModel``` class: 

```
def log_likelihood(self, candidate):
    alpha = 5 # preference for the original word compared to those with edit distance 1
    gamma = 7 # the higher the gamma >= 1, the more the model prefers the candidates with higher prior probability (i.e. lower edit distance)
    if candidate == self.surface_word:
        return alpha # more preference for the original word
    return -math.log(damerau_levenshtein(candidate, self.surface_word)) * gamma
```

### alpha

If ```alpha``` is too low, our approach might overcorrect the sentence.

For example, when ```alpha = 1``` and ```gamma = 8```, we have
> Input: Where should we meat tommorrow?
>
> Output: Where should be met tomorrow?

Modified to ```alpha = 6```, we have 
> Input: Where should we meat tommorrow?
>
> Output: Where should we meet tomorrow?

In this example, the overcorrection of the word "we" is alleviated by setting a higher ```alpha```. 

However, when ```alpha``` is too high, this approach has a tendency to preserve the original word.

For example, when ```alpha = 7```, we have
> Input: The plane tickets is expensive.
>
> Output: The plane tickets is expensive.

Modified to ```alpha = 3```, we have
> Input: The plane tickets is expensive.
>
> Output: The plane ticket is expensive.

Therefore we need to find an appropriate ```alpha```.

### gamma

The higher the ```gamma``` is, the more this approach prefers candidates with lower edit distance.

For example, when ```alpha = 5``` and ```gamma = 1```, we have
> Input: That is so god!
>
> Output: That is so cool!

Modified to ```gamma = 7```, we have
> Input: That is so god!
>
> Output: That is so good!

### N

Another hyperparameter that might also help is the edit distance threshold ```N``` during candidate selection: 

```
def get_suggestions_and_priors(self, word, masked_sentence, top_k=100, N=4)
```

We can filter out candidates s.t. the edit distance is higher than ```N=4```.



