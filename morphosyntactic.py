import sys

CONSOLE_WIDTH = 80
DICT_LEN = 4811854


def progress_bar(console_width=CONSOLE_WIDTH):
    """Draw console progress bar"""
    already_printed = 0
    print(" " * (console_width - 1), "|", sep="")
    print(" " * (console_width - 1), "|", "\b" * console_width, sep="", end="")

    def print_progress(progress):
        """Draw one character into progress bar if needed"""
        nonlocal already_printed
        if progress >= already_printed / console_width:
            print("=", end='')
            sys.stdout.flush()
            already_printed += 1
    return print_progress


class Morphosyntactic:
    """Stores morphosyntactic dictionary"""
    def __init__(self, dictionary_file_path):
        self.morphosyntactic_dictionary = {}
        self.dictionary_file_path = dictionary_file_path

    def create_morphosyntactic_dictionary(self):
        """Creates dictionary representation from file"""
        with open(self.dictionary_file_path, 'r', encoding='utf-8') as file:
            print("Tworzenie słownika morfosyntaktycznego:")
            print_progress = progress_bar()
            for line_number, line in enumerate(file.readlines()):
                if line_number % 1000 == 0:
                    progress = line_number / DICT_LEN
                    print_progress(progress)
                base_word, word, tags = line.rstrip("\n").split(";")
                if word.lower() in self.morphosyntactic_dictionary:
                    self.morphosyntactic_dictionary[word.lower()].append(base_word)
                else:
                    self.morphosyntactic_dictionary[word.lower()] = [base_word]
            print("\n")
        return self.morphosyntactic_dictionary

    def get_dictionary(self):
        if len(self.morphosyntactic_dictionary) == 0:
            self.create_morphosyntactic_dictionary()
        return self.morphosyntactic_dictionary


if __name__ == "__main__":
    morph = Morphosyntactic("data/polimorfologik-2.1.txt")
    morph.create_morphosyntactic_dictionary()
    d = morph.morphosyntactic_dictionary
    print(d["pić"])
    print(d["piła"])
    print(d["picie"])
    assert d["pić"] == [('pić', 'picie', ('subst:pl:gen:n2',)),
                        ('pić', 'pić', ('verb:inf:imperf:refl.nonrefl',))]
    assert d["picie"] == [('Picie', 'PIT', ('subst:sg:loc:m3', 'subst:sg:voc:m3')),
                          ('picie', 'picie', ('subst:sg:acc:n2', 'subst:sg:nom:n2', 'subst:sg:voc:n2')),
                          ('picie', 'pita', ('subst:sg:dat:f', 'subst:sg:loc:f')),
                          ('picie', 'pić', ('ger:sg:nom.acc:n2:imperf:aff:refl.nonrefl',))]
