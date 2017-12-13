from tokenization import tokenize


def load_dialogues_from_file(document_path, *,
                             do_tokenization=True,
                             remove_authors=False):
    with open(document_path) as file:
        lines = file.readlines()
        dialogues_list = [line for line in lines if not line.startswith("#")]
        dialogues_list = "".join(dialogues_list)
        dialogues_list = dialogues_list.split("\n\n")
        if remove_authors:
            dialogues_list = ["\n".join([line.split(":")[-1]
                                         for line in dialogue.split("\n")])
                              for dialogue in dialogues_list]
        if do_tokenization:
            dialogues_list = [tokenize(line) for line in dialogues_list]
    return dialogues_list


def split_dialogue(dialogue):
    dialogue_list = ([line.split(":")[-1] for line in dialogue.split("\n")])
    return dialogue_list


if __name__ == "__main__":
    dialogues = load_dialogues_from_file("data/drama_quotes_longer.txt")
    print(dialogues[:10])
    dialogues = load_dialogues_from_file("data/drama_quotes_longer.txt",
                                         do_tokenization=False)
    print(dialogues[:10])
    dialogues = load_dialogues_from_file("data/drama_quotes_longer.txt",
                                         remove_authors=True)
    print(dialogues[:10])
    dialogues = load_dialogues_from_file("data/drama_quotes_longer.txt",
                                         do_tokenization=False, remove_authors=True)
    print(dialogues[:10])