import pickle
from collections import defaultdict
from math import log
from os.path import isfile

from dialogue_load import load_list_of_dialogues_from_file
from morphosyntactic import Morphosyntactic
from tokenization import tokenize


class TF_IDF:
    def __init__(self, document_path: str, morphosyntactic: Morphosyntactic = None):
        self.document_path = document_path
        self.file_extension = ".tfidf"
        self.morphosyntactic = morphosyntactic
        self.dialogues_list = None
        self.tf_idf = {}
        self.term_frequency = defaultdict(lambda: defaultdict(lambda: 0))
        self.document_frequency = defaultdict(lambda: 0)

    def get_dialogues_list(self):
        if self.dialogues_list is None:
            self.dialogues_list = load_list_of_dialogues_from_file(self.document_path, remove_authors=True)
        return self.dialogues_list

    def save(self):
        if len(self.tf_idf) == 0:
            self.compute()
        path = self.document_path + self.file_extension
        with open(path, "wb") as tfidf_file:
            pickle.dump(self.tf_idf, tfidf_file)
        return path

    def load(self):
        path = self.document_path + self.file_extension
        if isfile(path):
            with open(self.document_path + self.file_extension, "rb") as tfidf_file:
                self.tf_idf = pickle.load(tfidf_file)
        else:
            self.compute()
            self.save()
        return self.tf_idf

    def _increase_tf(self, term, document_idx):
        if term in self.morphosyntactic.get_dictionary():
            base_forms = self.morphosyntactic.get_dictionary()[term]
            for base_form in base_forms:
                self.term_frequency[document_idx][base_form] += 1

    def _df_increaser(self):
        used_terms = []

        def increase_df_once(term):
            nonlocal used_terms
            if term in self.morphosyntactic.get_dictionary():
                base_forms = self.morphosyntactic.get_dictionary()[term]
                for base_form in base_forms:
                    if base_form not in used_terms:
                        self.document_frequency[base_form] += 1
                        used_terms.append(base_form)

        return increase_df_once

    def compute(self, text=None):
        if text is not None:
            dialogues = text
        else:
            dialogues = self.get_dialogues_list()

        for document_idx, document in enumerate(dialogues):
            increase_df = self._df_increaser()
            for term in document:
                term = term.lower()
                if term.isalpha():
                    self._increase_tf(term, document_idx)
                    increase_df(term)
        idf = {term: log(len(dialogues) / frequency, 2)
               for term, frequency in self.document_frequency.items()}
        for document_idx in self.term_frequency.keys():
            self.tf_idf[document_idx] = {term: idf[term] * self.term_frequency[document_idx][term]
                                         for term in
                                         set.intersection(set(self.term_frequency[document_idx].keys()),
                                                          set(idf.keys()))}
        return self.tf_idf


if __name__ == "__main__":
    text = ["Ktoś:Witajcie, ludzie!",
            "Ktoś:Przybywamy do Was w pokoju i dobrej woli!",
            "Ktoś:Robot(człowiek) nie może skrzywdzić człowieka, ani przez zaniechanie dopuścić, by doznał on (człowiek) krzywdy.",
            "Ktoś:Roboty widziały rzeczy, o których Wam, ludziom, się nie śniło.",
            "Ktoś:Roboty to Twoi plastikowi kumple, z którymi fajnie jest przebywać.",
            "Ktoś:Roboty mają lśniące, metalowe tyłki, których nie należy gryźć.",
            "Ktoś:I mają \nKtoś inny:plan."]

    text = [tokenize(phrase) for phrase in text]

    morph = Morphosyntactic("data/polimorfologik-2.1.txt")
    tfidf = TF_IDF("data/tf_idf_test", morph)
    tfidf.compute(text)

    print("term_frequency", tfidf.term_frequency)
    print("document_frequency", tfidf.document_frequency)
    print("tf_idf", tfidf.tf_idf)

    saved_path = tfidf.save()
    del tfidf
    tf_idf = TF_IDF("data/tf_idf_test", morph).load()
    print(tf_idf)

    tf_idf = TF_IDF("data/drama_quotes_test.txt", morph)
    tf_idf.compute()

    print("term_frequency", tf_idf.term_frequency)
    print("document_frequency", tf_idf.document_frequency)
    print("tf_idf", tf_idf.tf_idf)
    assert "malarz" in tf_idf.document_frequency
    assert "akwarelista" not in tf_idf.document_frequency
