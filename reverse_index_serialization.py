import os.path
import shelve
from enum import Enum


class IndexType(Enum):
    REVERSE = 1
    POSITIONAL = 2
    POSITIONAL_WITHOUT_LEMATIZATION = 3
    POSITIONAL_ENCODED = 4
    TITLE_4GRAM = 5


EXTENSIONS = {
    IndexType.REVERSE: ".rvi",
    IndexType.POSITIONAL: ".pos",
    IndexType.POSITIONAL_WITHOUT_LEMATIZATION: ".pwl",
    IndexType.POSITIONAL_ENCODED: ".enc",
    IndexType.TITLE_4GRAM: ".4gr"
}


def store_reverse_index(collection_path, create_reverse_index_function, index_creation_arguments=(),
                        index_type=IndexType.REVERSE):
    index = create_reverse_index_function(collection_path, *index_creation_arguments)
    stored_index = shelve.open(collection_path + EXTENSIONS[index_type])
    documents_beginnings = shelve.open(collection_path + EXTENSIONS[index_type] + ".beg")
    stored_index.update(index[0])
    documents_beginnings.update({"1": index[1]})
    return stored_index


def load_reverse_index(collection_path, index_type=IndexType.REVERSE):
    stored_index = shelve.open(collection_path + EXTENSIONS[index_type])
    return stored_index


def reverse_index_created(collection_path, index_type=IndexType.REVERSE):
    return os.path.isfile(collection_path + EXTENSIONS[index_type])
