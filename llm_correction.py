import enchant
import torch
from transformers import pipeline, BertTokenizer, logging, AutoTokenizer, AutoModelForCausalLM
from noisy_channel_model import NoisyChannelModel
from textdistance import damerau_levenshtein
import re


class BERTAutoCorrector():
    def __init__(self, unmasker_model_name, critic_model_name="GPT-2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.unmasker = pipeline('fill-mask', model=unmasker_model_name, device=self.device) 
        self.vocab = BertTokenizer.from_pretrained(unmasker_model_name).vocab
        
        """
        The critic was used to compute the log probability of a given sentence.
        This feature is not used in our implementation.
        """
        # self.critic_tokenizer = AutoTokenizer.from_pretrained(critic_model_name)
        # self.critic = AutoModelForCausalLM.from_pretrained(critic_model_name)
        # self.critic.to(self.device)
        # self.critic.eval()

        self.d = enchant.Dict("en_US")

    """
    Get the list of suggestions and their prior probability from the enchant library and the BERT model.
    The suggestions with edit distance >= N are filtered out.
    Input: word (str), masked_sentence (str), top_k (int), N (int)
    Output: suggestions (list), priors (dict)
    Example: get_suggestions_and_priors("hep", "I need [MASK] with my homework.", top_k=100)
    """
    def get_suggestions_and_priors(self, word, masked_sentence, top_k=100, N=4):
        priors = {}

        # Get suggestions from the enchant library
        raw_suggestions1 = self.d.suggest(word) + self.d.suggest(word.capitalize())
        raw_suggestions1 = list(filter(lambda suggestion: damerau_levenshtein(suggestion, word) <= N, raw_suggestions1))
        raw_suggestions1 = list({suggestion.lower() for suggestion in raw_suggestions1 if suggestion.lower() in self.vocab})
        if raw_suggestions1:
            raw_priors1 = self.unmasker(masked_sentence, targets=raw_suggestions1)
            suggestions1 = []
            for item in raw_priors1:
                suggestions1.append(item['token_str'])
                priors[item['token_str']] = item['score']
        else:
            suggestions1 = []

        # Get top K predictions from the BERT model
        predictions = self.unmasker(masked_sentence, top_k=top_k)
        raw_suggestions2 = [(prediction['token_str'], prediction['score']) for prediction in predictions]
        raw_suggestions2 = list(filter(lambda suggestion: damerau_levenshtein(suggestion[0], word) <= N, raw_suggestions2))
        suggestions2 = [suggestion for suggestion, _ in raw_suggestions2]
        for suggestion, score in raw_suggestions2:
            priors[suggestion] = score

        suggestions = list(set(suggestions1 + suggestions2)) # merge the suggestions from both sources
        # add the original word to the suggestions if it is in the vocab yet not already present in the suggestions
        if word not in suggestions and word in self.vocab and self.d.check(word):
            suggestions.append(word) 
            priors[word] = self.unmasker(masked_sentence, targets=[word])[0]['score']

        return suggestions, priors

    """
    Get the corrected sentence by replacing the masked word with the corrected word through the noisy channel model.
    Input: word (str), masked_sentence (str)
    Output: corrected_sentence (str)
    Example: correct("hep", "I need [MASK] with my homework.")
    Note: The input sentence should contain the masked word '[MASK]'.
    """
    def correct(self, word, masked_sentence):
        surface_word = word.lower()
        suggestions, priors = self.get_suggestions_and_priors(surface_word, masked_sentence)
        if len(suggestions) == 0:
            return masked_sentence.replace('[MASK]', word)
        noisy_channel_model = NoisyChannelModel(suggestions, priors, surface_word)
        corrected_word = noisy_channel_model.correct()
        if word.istitle():
            corrected_word = corrected_word.capitalize() # simply preserve the capitalization
        return masked_sentence.replace('[MASK]', corrected_word)

    """
    Use GPT-2 Model to compute the log probability of a given sentence.
    Input: sentence (str)
    Output: sentence_log_prob (float)
    """
    def get_sentence_score(self, sentence):
        input_ids = self.critic_tokenizer.encode(sentence, return_tensors="pt").to(self.device) # encode input sentence
        output = self.critic(input_ids=input_ids)
        logits = output.logits # raw output of shape (1, num_of_tokens, vocab_size)
        probs = torch.nn.functional.softmax(logits, dim=-1) # convert logits to probabilities
        log_probs = torch.log(probs) # convert probabilities to log probabilities
        sentence_log_prob = log_probs[0, torch.arange(input_ids.size(1)), input_ids.squeeze(0)].sum() # compute the log probability of the input sentence
        return sentence_log_prob.item()

    """
    Get the list of possible misspelled words in the input sentence.
    Input: sentence (str)
    Output: wrong_words (list)
    Example: identidy_typos("I cant bore you anymore.") should return [("cant", 1), ("bore", 2)]
    """
    def identify_misspelled_words(self, sentence):
        wrong_words = []
        words = sentence.split()
        words = [re.sub(r'[.,!?;\']', '', word) for word in words]
        for i, word in enumerate(words):
            if not (self.d.check(word) and word.lower() in self.vocab):
                wrong_words.append((word.lower(), i))
        return wrong_words

    """
    Correct identified typos from left to right in the input sentence.
    The misspelled words are corrected first.
    """
    def auto_correct(self, sentence):
        def mask(sentence, i):
            words = sentence.split()
            if i == len(words) - 1 and sentence[-1] in [',', '.', '!', '?', ';']:
                words[i] = '[MASK]' + sentence[-1] # preserve the punctuation
            else:
                words[i] = '[MASK]'
            return ' '.join(words)
        
        sentence = re.sub(r'[\']', '', sentence) # remove apostrophes in the sentence
        corrected_sentence = sentence
        wrong_words = self.identify_misspelled_words(corrected_sentence)
        # print("Wrong_words: ", wrong_words)
        for word, i in wrong_words:
            masked_sentence = mask(corrected_sentence, i)
            # print("Masked sentence: ", masked_sentence)
            # print("word: ", word)
            corrected_sentence = self.correct(word, masked_sentence)
            # print("Corrected sentence: ", corrected_sentence)
        # correct the remaining words
        words = sentence.split()
        words = [(re.sub(r'[.,!?;\']', '', word), i) for i, word in enumerate(words)]
        for word, i in words:
            masked_sentence = mask(corrected_sentence, i)
            # print("Masked sentence: ", masked_sentence)
            # print("word: ", word)
            corrected_sentence = self.correct(word, masked_sentence)
            # print("Corrected sentence: ", corrected_sentence)
        return corrected_sentence
        

if __name__ == '__main__':
    import argparse
    import sys

    # Get the input sentence from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input sentence to be auto-corrected')
    parser.add_argument('--debug', action='store_true', help='whether to displace warnings / error messages')
    parser.add_argument('--dataset_dir', type=str, help="directory to the input dataset (.csv)")
    parser.add_argument('--output_dir', type=str, help="directory to the output (.csv)")
    args = parser.parse_args()
    input_sentence = args.input

    # Filter warnings and errors if they are not necessary
    ignore_warnings = not args.debug
    if ignore_warnings:
        class Suppressor:
            def write(self, *args, **kwargs):
                pass
        sys.stderr = Suppressor()
        logging.set_verbosity_error()

    # Load the BERT-Large-uncased model for masked language modeling
    unmasker_model_name = "bert-large-uncased"

    auto_corrector = BERTAutoCorrector(unmasker_model_name)

    # Correct the input sentence given from the command line
    if input_sentence:
        print('Input sentence:', input_sentence)
        corrected_sentence = auto_corrector.auto_correct(input_sentence)
        print('Corrected sentence:', corrected_sentence)

    # Correct the sentences in a dataset (.csv file) and save the output to another file
    if args.dataset_dir:
        if args.output_dir is None:
            print("Please provide the output directory.")
            exit()

        import pandas as pd

        # Load the test data
        test_data = pd.read_csv('data.csv')
        input_sentences = test_data['Original Sentence'].tolist()
        target_sentences = test_data['Target Sentence'].tolist()

        corrected_sentences = []
        edit_distance = []

        for i, sentence in enumerate(input_sentences):
            corrected_sentence = auto_corrector.auto_correct(sentence)
            corrected_sentences.append(corrected_sentence)
            edit_distance.append(damerau_levenshtein(corrected_sentence, target_sentences[i]))
            print(f"Processed {i + 1}/{len(input_sentences)} sentences.")

        test_data['Corrected Sentence'] = corrected_sentences
        test_data['Edit Distance'] = edit_distance

        test_data.to_csv(args.output_dir, index=False)
