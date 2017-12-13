import random
from collections import defaultdict
from typing import List, Dict, Set

from tf_idf import TF_IDF
from tokenization import tokenize, tokenize_dialogue
import morphosyntactic as morph
from dialogue_load import load_dialogues_from_file, split_dialogue
from reverse_index_serialization import load_reverse_index, reverse_index_created, store_reverse_index, IndexType


def create_reverse_index(path_to_documents_collection, morphosyntactic):
    index = defaultdict(lambda: set())

    with open(path_to_documents_collection, 'r', encoding='utf-8') as file:
        print("+++ creating reverse index +++")
        for line_number, line in enumerate(file):
            if line.startswith("#"):
                continue
            line = tokenize(line.split(":")[-1])
            for token in line:
                base_tokens = morphosyntactic.get_dictionary().get(token, [])
                for base_token in base_tokens:
                    index[base_token].add(line_number)
        print("+++ reverse index created +++")
        print(len(index))
    return [index, []]


def create_dialogue_reverse_index(path_to_documents_collection, morphosyntactic):
    index = defaultdict(lambda: set())

    dialogues = load_dialogues_from_file(path_to_documents_collection,
                                         remove_authors=True, do_tokenization=True)
    print("+++ creating reverse index +++")
    for dialogue_idx, dialogue in enumerate(dialogues):
        for token in dialogue:
            base_tokens = morphosyntactic.get_dictionary().get(token, [])
            for base_token in base_tokens:
                index[base_token].add(dialogue_idx)
    print("+++ reverse index created +++")
    print(len(index))
    return [index, []]


def weighted_draw(possible_quotes):
    total = sum(w for c, w in possible_quotes)
    r = random.uniform(0, total)
    upto = 0
    for c, w in possible_quotes:
        if upto + w >= r:
            return c, w
        upto += w


class MCR:
    def __init__(self, *,
                 morphosyntactic_path="data/polimorfologik-2.1.txt",
                 quotes_path="data/drama_quotes_longer.txt",
                 filter_rare_results=False):
        self.morphosyntactic = morph.Morphosyntactic(morphosyntactic_path)
        self.morphosyntactic.create_morphosyntactic_dictionary()
        self.stopwords = MCR.load_stopwords()
        self.index = self.load_index(quotes_path)
        self.quotes: List[str] = load_dialogues_from_file(quotes_path, do_tokenization=False, remove_authors=False)
        tf_idf_generator = TF_IDF(quotes_path, self.morphosyntactic)
        self.tf_idf: Dict[int, Dict[str, float]] = tf_idf_generator.load()
        self.filter_rare_results = filter_rare_results

        self.randomized = None
        self.default_quote = None
        self.used_quotes: Set[str] = None

    @staticmethod
    def load_stopwords():
        try:
            with open("data/stopwords.txt") as file:
                line = file.readline()
                stopwords = line.split(", ")
        except FileNotFoundError:
            stopwords = ()
        return stopwords

    def load_index(self, quotes_path):
        if reverse_index_created(quotes_path, IndexType.DIALOGUE):
            index = load_reverse_index(quotes_path, IndexType.DIALOGUE)
        else:
            index = store_reverse_index(quotes_path, create_dialogue_reverse_index, [self.morphosyntactic],
                                        index_type=IndexType.DIALOGUE)
        return index

    def run(self, *, randomized=True, default_quote="Jeden rabin powie tak, a inny powie nie."):
        self.randomized = randomized
        self.default_quote = default_quote
        self.used_quotes = {""}
        try:
            while True:
                line = self.get_tokenized_line()
                results = self.find_matching_quotes(line)
                selected_quote = self.select_quote(results, line)
                self.used_quotes.add(selected_quote)
                print(selected_quote)
        except KeyboardInterrupt:
            return
        except EOFError:
            return

    def get_tokenized_line(self):
        line = input("> ").strip()
        if len(line) > 0 and line[0].upper():
            line = line[0].lower() + line[1:]

        line = list(filter(lambda x: x not in self.stopwords, tokenize(line)))
        line = [self.morphosyntactic.get_dictionary().get(token, []) for token in line]
        return line

    def find_matching_quotes(self, line):
        quotes_sets = []
        for base_tokens in line:
            quotes_indices = set()
            for base_token in base_tokens:
                quotes_indices.update(self.index.get(base_token, []))
            quotes_sets.append(quotes_indices)
        results = defaultdict(lambda: set())
        for i, quotes_set in enumerate(quotes_sets):
            for quote_number in quotes_set:
                results[quote_number].add(i)
        return results

    def select_quote(self, results: Dict[int, Set[int]], line: List[List[str]]):
        if len(results) == 0:
            return self.default_quote
        if self.filter_rare_results:
            if any((len(k) > 1 for k in results.values())):
                results = {k: v for k, v in results.items() if len(v) > 1}

        possible_quotes = self._get_quotes_from_indices(results)
        for possible_quote in possible_quotes:
            possible_quote[0], possible_quote[1] = self.evaluate_quote(possible_quote, line)  # TODO: Select best quote

        if self.randomized:
            selected_quote = self._select_randomized_quote(possible_quotes)
        else:
            selected_quote = self._select_best_quote(possible_quotes)
        return selected_quote

    def _get_quotes_from_indices(self, results):
        possible_quotes = []
        for result in results.keys():
            try:
                possible_quotes.append([self.quotes[result], result])
            except IndexError:
                pass
        return possible_quotes

    def _select_randomized_quote(self, possible_quotes):
        selected_quote = [""]
        while selected_quote[0] in self.used_quotes:
            if len(possible_quotes) == 0:
                return self.default_quote
            selected_quote = weighted_draw(possible_quotes)
            possible_quotes.remove(list(selected_quote))
        return selected_quote[0]

    def _select_best_quote(self, possible_quotes):
        max_value = max(possible_quotes, key=lambda x: x[1])[1]
        possible_quotes_max = list(filter(lambda x: x[1] == max_value, possible_quotes))
        selected_quote = possible_quotes_max[0]
        i = 0
        while selected_quote[0] in self.used_quotes:
            i += 1
            if i < len(possible_quotes):
                selected_quote = possible_quotes[i]
            else:
                selected_quote = possible_quotes[0]
                break
        return selected_quote[0]

    def evaluate_quote(self, quote, question, choose_answer=False):
        def score_function(word):
            try:
                word_score = self.tf_idf[quote_idx][word]
            except KeyError:
                word_score = 0
            return word_score

        quote_idx = quote[1]
        raw_quote_text = split_dialogue(quote[0])
        quote_text = tokenize_dialogue(quote[0])
        for dialogue_idx, dialogue in enumerate(quote_text):
            quote_text[dialogue_idx] = [self.morphosyntactic.get_dictionary().get(token, []) for token in dialogue]

        best_quote = raw_quote_text[0]
        cosine = 0
        question_vector = WordVector(question, score_function)

        for dialogue_idx, dialogue in enumerate(quote_text):
            quote_slice = quote_text[:dialogue_idx + 1]
            quote_slice = [item for sublist in quote_slice for item in sublist]
            quote_vector = WordVector(quote_slice, score_function)

            try:
                new_cosine = (quote_vector @ question_vector) / (quote_vector.len() * question_vector.len())
            except ZeroDivisionError:  # TODO: Check tf-idf
                new_cosine = 0
            if new_cosine > cosine and (not choose_answer or len(quote_text) > dialogue_idx + 1):
                cosine = new_cosine
                best_quote = raw_quote_text[dialogue_idx + choose_answer]

        return best_quote, cosine


class WordVector:
    def __init__(self, quote, score_function):
        self.vector = defaultdict(lambda: 0)
        for possible_words in quote:
            for base_word in possible_words:
                base_word_score = score_function(base_word)
                self.vector[base_word] = base_word_score

    def len(self):
        length = sum(value ** 2 for value in self.vector.values())
        return length

    def __getitem__(self, item):
        return self.vector[item]

    def __matmul__(self, other):
        dot_product = sum(self[word] * other[word]
                           for word in set.union(set(self.vector.keys()), set(other.vector.keys())))
        return dot_product

    def __str__(self):
        return str(self.vector)


if __name__ == "__main__":
    MCR().run()
