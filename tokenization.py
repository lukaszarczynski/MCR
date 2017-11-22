import re


def tokenize(text: str):
    """Split text into list of alphanumeric words and other characters"""
    tokenized = re.split('(\w+)', text)
    return [word for word in tokenized if word != '']


if __name__ == "__main__":
    print(tokenize("lorem, ipsum"))
    assert tokenize("lorem, ipsum") == ['lorem', ', ', 'ipsum']
    test_text = input()
    print(tokenize(test_text))
