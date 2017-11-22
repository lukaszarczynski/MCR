import random
from collections import defaultdict
from math import sqrt

from tf_idf import TF_IDF
from tokenization import tokenize
import morphosyntactic as morph
from dialogue_load import load_list_of_dialogues_from_file
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

    dialogues = load_list_of_dialogues_from_file(path_to_documents_collection,
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

# def quotes_split_exists(split_quotes_path):
#     return os.path.isfile(split_quotes_path)


# def split_quotes(collection_path, split_quotes_path):
#     with open(collection_path) as file:
#         with open(split_quotes_path, "w") as new_file:
#             for line in file:
#                 for quote in line.split(" . "):
#                     new_file.write(quote.rstrip("\n") + "\n")


def _evaluate_quote_old(quote, keywords, question):
    value = 1
    for keyword in keywords:
        value *= len(keywords) ** len(question[keyword][0])
    return value / sqrt(len(quote))


def evaluate_quote(quote_idx, question, tf_idf):
    total_score = 0
    for possible_words in question:
        max_word_score = 0
        for base_word in possible_words:
            try:
                base_word_score = tf_idf[quote_idx][base_word]
            except KeyError:
                base_word_score = 0
            max_word_score = max(max_word_score, base_word_score)
        total_score += max_word_score
    return total_score


def weighted_draw(possible_quotes):
    total = sum(w for c, w in possible_quotes)
    r = random.uniform(0, total)
    upto = 0
    for c, w in possible_quotes:
        if upto + w >= r:
            return c, w
        upto += w


# collection_path = "data/tokenized_quotes.txt"
quotes_path = "data/drama_quotes_longer.txt"
default_quote = "Jeden rabin powie tak, a inny powie nie."
randomized = True


def mcr():
    morphosyntactic = morph.Morphosyntactic("data/polimorfologik-2.1.txt")
    morphosyntactic.create_morphosyntactic_dictionary()

    # if not quotes_split_exists(quotes_path):
    #     split_quotes(collection_path, quotes_path)

    if reverse_index_created(quotes_path, IndexType.DIALOGUE):
        index = load_reverse_index(quotes_path, IndexType.DIALOGUE)
    else:
        index = store_reverse_index(quotes_path, create_dialogue_reverse_index, [morphosyntactic],
                                    index_type=IndexType.DIALOGUE)

    with open("data/stopwords.txt") as file:
        line = file.readline()
        stopwords = line.split(", ")

    quotes = load_list_of_dialogues_from_file(quotes_path, do_tokenization=False, remove_authors=False)

    print("tf-idf started")
    tf_idf_generator = TF_IDF(quotes_path, morphosyntactic)
    tf_idf = tf_idf_generator.load()
    print("tf-idf done")

    used_quotes = [""]
    try:
        while True:
            line = input("> ").strip()
            if len(line) > 0 and line[0].upper():
                line = line[0].lower() + line[1:]

            line = list(filter(lambda x: x not in stopwords, tokenize(line)))
            line = [morphosyntactic.get_dictionary().get(token, []) for token in line]

            quotes_sets = []
            for base_tokens in line:
                quotes_indices = set()
                for base_token in base_tokens:
                    quotes_indices.update(index.get(base_token, []))
                quotes_sets.append(quotes_indices)
            results = defaultdict(lambda: set())
            for i, quotes_set in enumerate(quotes_sets):
                for quote_number in quotes_set:
                    results[quote_number].add(i)
            if len(results) == 0:
                print(default_quote, "\n")
            else:
                if any((len(k) > 1 for k in results.values())):
                    results = {k: v for k, v in results.items() if len(v) > 1}
                possible_quotes = []
                for result in results.keys():
                    try:
                        possible_quotes.append([quotes[result], result])
                    except IndexError:
                        pass
                for possible_quote in possible_quotes:
                    possible_quote[1] = evaluate_quote(possible_quote[1], line, tf_idf)
                if randomized:
                    selected_quote = [""]
                    while selected_quote[0] in used_quotes:
                        selected_quote = weighted_draw(possible_quotes)
                else:
                    max_value = max(possible_quotes, key=lambda x: x[1])[1]
                    possible_quotes_max = list(filter(lambda x: x[1] == max_value, possible_quotes))
                    selected_quote = possible_quotes_max[0]
                    i = 0
                    while selected_quote[0] in used_quotes:
                        i += 1
                        if i < len(possible_quotes):
                            selected_quote = possible_quotes[i]
                        else:
                            selected_quote = possible_quotes[0]
                            break
                print(selected_quote[0])
                used_quotes.append(selected_quote[0])
    except KeyboardInterrupt:
        return
    except EOFError:
        return


if __name__ == "__main__":
    mcr()
