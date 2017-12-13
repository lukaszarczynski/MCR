import re


def tokenize(text: str):
    """Split text into list of alphanumeric words and other characters"""
    tokenized = re.split('(\w+)', text)
    return [word for word in tokenized if word != '']


def tokenize_dialogue(dialogue_text: str):
    dialogue_list = ([line.split(":")[-1] for line in dialogue_text.split("\n")])
    dialogue_list = [tokenize(line.lower()) for line in dialogue_list]
    return dialogue_list


if __name__ == "__main__":
    print(tokenize("lorem, ipsum"))
    assert tokenize("lorem, ipsum") == ['lorem', ', ', 'ipsum']
    test_text = input()
    print(tokenize(test_text))
