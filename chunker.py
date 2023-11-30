import argparse
import os
import re
import warnings
from typing import List

import benepar
import spacy
from nltk.tree import Tree
from tqdm import tqdm
from transformers import logging as hf_logging

warnings.simplefilter("ignore")
hf_logging.set_verbosity_error()

# Load the English model for spaCy
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})


def format_chunked_list(splits: List[str]):
    """Formats the output of the chunker into a readable string.

    Args:
        splits (List[str]): List of strings to be formatted.

    Returns:
        str: Formatted string.

    Example:
    >>> format_chunked_list(["You ", "know ", "/ ", "/ ", "that ", "it's ", "bad", "."])
    "You know / that it's bad."
    """
    splits_ = []
    for e in splits:
        if e != "/ " or (splits_ and splits_[-1] != "/ "):
            splits_.append(e)
    return re.sub(r"(?<! )/ ", "", "".join(splits_))
    # return "".join(splits_)


def parse_tree(sentence: str):
    """Parse a sentence and return its parse tree and a token to token_ws mapping.

    Args:
        sentence (str): Sentence to be parsed.

    Returns:
       Tuple[str, dict]: Tuple of the parse tree and
       a dictionary mapping tokens to tokens with whitespace.

    Note:
    - Requires an NLP model (like spaCy) with dependency parsing capabilities.
    - The function is dependent on the model's ability to parse sentences
      and might yield different results with different models.
    """

    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    tree_string = sent._.parse_string

    token_space_mapping = {}
    for i, token in enumerate(doc):
        if token.text not in token_space_mapping.keys():
            token_space_mapping[token.text] = [token.text_with_ws]
        else:
            token_space_mapping[token.text].append(token.text_with_ws)
    return tree_string, token_space_mapping


def replace_special_tags_in_tree(tree: Tree):
    """Replace special tags in the tree with their original symbols.

    Args:
        tree (Tree): The tree to be processed.

    Returns:
        Tree: The processed tree.
    """
    replacements = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }

    for pos in range(len(tree)):
        if type(tree[pos]) is Tree:
            replace_special_tags_in_tree(tree[pos])
        elif tree[pos] in replacements:
            tree[pos] = replacements[tree[pos]]
    return tree


def merge_chunked_strings(list_of_sentence: List[str]):
    """
    Merge multiple chunked strings into a single string, preserving chunk divisions.

    Args:
    strings (list of str): The list of chunked strings to merge.

    Returns:
    str: The merged string with chunk divisions preserved.

    Example:
    >>> merge_chunked_strings(['He went to the store / to buy groceries.', 'He went / to the store to buy groceries.'])
    'He went / to the store / to buy groceries.'
    """
    words = [[] for _ in list_of_sentence]
    slash_positions = [[] for _ in list_of_sentence]
    for i, s in enumerate(list_of_sentence):
        words_with_slash = s.split()
        words[i] = []
        slash_positions[i] = []
        w_pre = ""
        for w in words_with_slash:
            if w != "/":
                words[i].append(w)
            if w_pre != "/" and w_pre != "":
                slash_positions[i].append(w == "/")
            w_pre = w
        slash_positions[i].append(False)

    merged = []
    slash_positions_merged = [any(elements) for elements in zip(*slash_positions)]
    for i in range(len(words[0])):
        merged.append(words[0][i])
        if slash_positions_merged[i]:
            merged.append("/")

    return " ".join(merged)


def split_sentence_rule1(sentence):
    """
    Split a sentence according to a specific rule involving conjunctions and relative pronouns.

    Rule:
    Split before a conjunction or a relative pronoun, except when it modifies a subject.

    Args:
    sentence (str): The sentence to split.

    Returns:
    str: The sentence split according to the specified rule.

    Example:
    >>> split_sentence_rule1("You interacted with the company that you work for or that you own.")
    'You interacted with the company / that you work for / or / that you own.'
    """

    def is_subject_np(subtree):
        """
        Check if the subtree is a Noun Phrase acting as a subject in the sentence.
        """
        if hasattr(subtree, "parent") and subtree.parent().label() == "S":
            for sibling in subtree.parent():
                if sibling != subtree and sibling.label() == "VP":
                    return True
        return False

    parsed_tree_string, token_space_mapping = parse_tree(sentence)
    parsed_tree = Tree.fromstring(parsed_tree_string)
    parsed_tree = replace_special_tags_in_tree(parsed_tree)

    def traverse_tree(tree):
        splits = []
        for subtree in tree:
            if isinstance(subtree, Tree):
                if subtree.label() == "CC" and subtree[0] != ",":
                    splits.append("/ ")
                elif subtree.label() == "SBAR" and not is_subject_np(subtree.parent()):
                    splits.append("/ ")
                splits.extend(traverse_tree(subtree))
            else:
                token_text = token_space_mapping.get(subtree).pop(0)
                splits.append(token_text)
        return splits

    def modifies_subject(tree):
        """
        Check if the given tree modifies a subject.
        """
        if tree.parent() is None:
            return False
        parent_label = tree.parent().label()
        if parent_label in ["S"]:
            return True
        return False

    # Enhance the tree with parent pointers.
    for subtree in parsed_tree.subtrees():
        for child in subtree:
            if isinstance(child, Tree):
                child.set_label(child.label())
                child.parent = lambda s=subtree: s
    return format_chunked_list(traverse_tree(parsed_tree))


def split_sentence_rule2(sentence):
    """
    Split a sentence according to a specific rule involving to-infinitives, prepositions, or gerunds.

    Rule:
    Split before a to-infinitive, a preposition, or a gerund followed by three or more words.

    Args:
    sentence (str): The sentence to split.

    Returns:
    str: The sentence split according to the specified rule.

    Example:
    >>> split_sentence_rule2("I've just finished cleaning up my room.")
    "I've just finished / cleaning up my room."

    Note:
    - Requires an NLP model (like spaCy) for part-of-speech tagging.
    - The function is dependent on the model's accuracy of the POS tagging
      and might yield different results with different models.
    """

    def word_to_split(word):
        # preposition
        if word.pos_ == "ADP":
            return True
        # to-infinitive
        if word.pos_ == "PART" and word.text == "to":
            return True
        # gerund
        if word.tag_ == "VBG" and "ing" in word.text:
            return True
        return False

    doc = nlp(sentence)
    words = list(doc)

    splits = []
    count = 0

    for i in range(len(words) - 1, -1, -1):
        word = words[i]
        if word.text == ".":
            count -= 1
        count += 1
        if word_to_split(word) and count > 3:
            splits.insert(0, word.text_with_ws)
            splits.insert(0, "/ ")
            count = 0
        else:
            splits.insert(0, word.text_with_ws)

    return format_chunked_list(splits)


def split_sentence_rule2_chunked_input(sentence_with_slash):
    """
    Split a chunked sentence further based on specific grammatical rules.

    Rule:
    Split before a to-infinitive, a preposition, or a gerund followed by three or more words.

    Args:
    sentence_with_slash (str): The chunked sentence to split further, with existing chunks indicated by slashes.

    Returns:
    str: The sentence further split according to the specified rule.

    Example:
    >>> split_sentence_rule2_chunked_input("Self-interest isn't a narrowly defined concept just for your immediate utility.")
    "Self-interest isn't a narrowly defined concept just / for your immediate utility."
    >>> split_sentence_rule2_chunked_input("Self-interest / isn't / a narrowly defined concept just / for your immediate utility.")
    "Self-interest / isn't / a narrowly defined concept just / for your immediate utility."

    Note:
    - Requires an NLP model (like spaCy) for part-of-speech tagging.
    - The function handles sentences that have already been chunked and further splits them according to the specified rule.
    """

    def word_to_split(word):
        # preposition
        if word.pos_ == "ADP":
            return True
        # to-infinitive
        if word.pos_ == "PART" and word.text == "to":
            return True
        # gerund
        if word.tag_ == "VBG" and "ing" in word.text:
            return True
        return False

    # tokenize each word by SpaCy
    words_with_slash = sentence_with_slash.split()
    words_with_slash_spacy = []
    for word in words_with_slash:
        doc = nlp(word)
        words_with_slash_spacy.extend([token.text for token in doc])

    # get slash_positions (e.g. [False, False, True, False, ...])
    words = []
    slash_positions = []
    w_pre = ""
    for w in words_with_slash_spacy:
        if w != "/":
            words.append(w)
        if w_pre != "/" and w_pre != "":
            slash_positions.append(w == "/")
        w_pre = w
    slash_positions.append(False)

    doc = nlp(format_chunked_list(sentence_with_slash.replace("/ ", "")))
    words = list(doc)

    splits = []
    count = 0

    for i in range(len(words) - 1, -1, -1):
        word = words[i]
        if slash_positions[i]:
            count = 0
        if word.text == ".":
            count -= 1
        count += 1
        if word_to_split(word) and count > 3:
            splits.insert(0, word.text_with_ws)
            splits.insert(0, "/ ")
            count = 0
        else:
            splits.insert(0, word.text_with_ws)

    return merge_chunked_strings([sentence_with_slash, format_chunked_list(splits)])


def split_sentence_rule3(sentence):
    """
    Split a sentence after a long subject (three or more words).

    Rule:
    Split the sentence after a subject if the subject consists of three or more words.

    Args:
    sentence (str): The sentence to split.

    Returns:
    str: The sentence split after a long subject.

    Example:
    >>> split_sentence_rule3('The big brown dog barked loudly.')
    'The big brown dog / barked loudly.'

    Note:
    - Requires an NLP model (like spaCy) for dependency parsing.
    - The function is dependent on the model's ability to parse sentences
      and might yield different results with different models.
    """
    doc = nlp(sentence)
    splits = []
    current_split = 0

    for token in doc:
        if "nsubj" in token.dep_:
            subject_tokens = [t for t in token.subtree]
            if len(subject_tokens) >= 3 and token.text_with_ws != token.text:
                split_index = subject_tokens[-1].i + 1
                splits.append(doc[current_split:split_index].text)
                current_split = split_index
    splits.append(doc[current_split:].text)
    return " / ".join(splits)


def split_sentence_rule4(sentence):
    """
    Split a sentence before or after certain punctuation marks.

    Rule:
    Split before or after a comma (except one used for listing in a series of words),
    a semicolon, a hyphen, or other marks of punctuation

    Args:
    sentence (str): The sentence to split.

    Returns:
    str: The sentence split according to the specified punctuation rules.

    Example:
    >>> split_sentence_rule4("I like apples, oranges, and -- bananas; actually, I don't like - fruits.")
    "I like apples, oranges, and -- / bananas; / actually, I don't like - / fruits."
    """

    def is_subject_np(subtree):
        """
        Check if the subtree is a Noun Phrase acting as a subject in the sentence.
        """
        parent = subtree.parent()
        if parent and parent.label() == "S":
            for sibling in parent:
                if sibling != subtree and sibling.label() == "VP":
                    return True
        return False

    def check_words_listed(tree, i):
        """
        Check if comma is used for a list of series of words.
        """
        prev_comma = tree[i - 2]
        next_comma = tree[i]
        if next_comma.label() == "CC":
            next_comma = tree[i + 1]
        if len(prev_comma.leaves()) + len(next_comma.leaves()) >= 5:
            return False
        elif len(prev_comma.leaves()) + len(next_comma.leaves()) <= 2:
            # If the sum of the subtree lengths before and after the comma \
            # is less than 2, it is considered part of a word list
            return True
        else:
            # case of length 3~4; Returns True if both subtrees are "DT+NN or length 1"
            # breakpoint()
            return (
                len(prev_comma.leaves()) == 1
                or (prev_comma[0].label() == "DT" and prev_comma[1].label() in "NNS")
            ) and (
                len(next_comma.leaves()) == 1
                or (next_comma[0].label() == "DT" and next_comma[1].label() in "NNS")
            )

    parsed_tree_string, token_space_mapping = parse_tree(sentence)
    parsed_tree = Tree.fromstring(parsed_tree_string)
    parsed_tree = replace_special_tags_in_tree(parsed_tree)

    def traverse_tree(tree):
        splits = []
        for i, subtree in enumerate(tree):
            if isinstance(subtree, Tree):
                if i > 0:
                    if tree[i - 1].label() == ":":
                        splits.append("/ ")
                    elif (
                        i > 1
                        and tree[i - 1].label() == ","
                        and not check_words_listed(tree, i)
                    ):
                        splits.append("/ ")
                splits.extend(traverse_tree(subtree))
            else:
                token_text = token_space_mapping.get(subtree).pop(0)
                splits.append(token_text)
        if isinstance(subtree, Tree) and subtree.label() in [":", ","]:
            # Case where the rightmost child is a symbol
            splits.append("/ ")
        return splits

    # Enhance the tree with parent pointers.
    for subtree in parsed_tree.subtrees():
        for child in subtree:
            if isinstance(child, Tree):
                child.set_label(child.label())
                child.parent = lambda s=subtree: s

    return format_chunked_list(traverse_tree(parsed_tree))


def split_sentence_rule5(sentence):
    """
    Split a sentence based on the position of adverbial or prepositional phrases.

    Rule:
    Split after an adverbial or prepositional phrase that is at the beginning of
    a sentence or right after a subordinating conjunction or relative pronoun.

    Args:
    sentence (str): The sentence to split.

    Returns:
    str: The sentence split according to the specified rule.

    Example:
    >>> split_sentence_rule5("In fact this past weekend we had one of our trucks.")
    'In fact / this past weekend we had one of our trucks.'
    """

    def set_left_siblings(tree, left_siblings=[]):
        for child in tree:
            if isinstance(child, Tree):
                child.left_siblings = list(left_siblings)
                left_siblings.append(child)
                set_left_siblings(child, left_siblings=[])

    def is_subject_np(subtree):
        """
        Check if the subtree is a Noun Phrase acting as a subject in the sentence.
        """
        parent = subtree.parent()
        if parent and parent.label() == "S":
            for sibling in parent:
                if sibling != subtree and sibling.label() == "VP":
                    return True
        return False

    def is_clause_led_by_conjunction(subtree):
        """
        Check if the subtree is a clause led by a subordinating conjunction.
        """
        if len(subtree.left_siblings) == 0:
            return False
        return subtree.left_siblings[-1].label() == "CC"

    def is_clause_led_by_relative_pronoun(subtree):
        """
        Check if the subtree is a clause led by a relative pronoun.
        """
        return subtree.label() == "SBAR" and not is_subject_np(subtree.parent())

    parsed_tree_string, token_space_mapping = parse_tree(sentence)
    parsed_tree = Tree.fromstring(parsed_tree_string)
    parsed_tree = replace_special_tags_in_tree(parsed_tree)

    def traverse_tree(tree, depth):
        splits = []

        # is_bos to handle cases where PP or ADVP occur consecutively
        # at the beginning of a sentence
        is_bos = False
        if depth == 0:
            is_bos = True
        for i, subtree in enumerate(tree):
            if isinstance(subtree, Tree):
                splits.extend(traverse_tree(subtree, depth + 1))

                # case1: at the beginning of a sentence
                if is_bos and subtree.label() in ["PP", "ADVP"]:
                    splits.append("/ ")
                elif subtree.label() == ",":
                    if len(splits) > 1 and splits[-2] == "/ ":
                        splits[-1], splits[-2] = splits[-2], splits[-1]
                else:
                    is_bos = False

                    if subtree.label() in ["PP", "ADVP"] and (
                        subtree.left_siblings == []
                        or is_clause_led_by_conjunction(subtree)
                    ):
                        # case 2: right after conjunction
                        if is_clause_led_by_conjunction(
                            subtree
                        ) or is_clause_led_by_conjunction(tree):
                            splits.append("/ ")
                        # case 3: right after relative pronoun
                        if (
                            tree.label() == "S"
                            and hasattr(tree, "parent")
                            and is_clause_led_by_relative_pronoun(tree.parent())
                        ):
                            splits.append("/ ")
            else:
                token_text = token_space_mapping.get(subtree).pop(0)
                splits.append(token_text)
        return splits

    # Add left siblings of subtrees.
    set_left_siblings(parsed_tree)

    # Enhance the tree with parent pointers.
    for subtree in parsed_tree.subtrees():
        for child in subtree:
            if isinstance(child, Tree):
                child.set_label(child.label())
                child.parent = lambda s=subtree: s

    return format_chunked_list(traverse_tree(parsed_tree, depth=0))


def split_sentence_by_five_rules(sentence):
    """
    Apply five different rules to split a sentence and merge the results.

    Args:
    sentence (str): The sentence to split.

    Returns:
    str: The sentence split according to five different rules.

    Example:
    >>> split_sentence_by_five_rules("The quick brown fox that jumps over the lazy dog is very agile, and he and his brother often plays in the fields.")
    'The quick brown fox that jumps / over the lazy dog / is very agile, / and he / and his brother / often plays in the fields.'

    Note:
    - The function sequentially applies five different sentence splitting rules.
    - The first step involves merging the results of rules 1, 3, 4, and 5.
    - The second step applies rule 2 to the merged chunked string.
    """

    # Rule 1, 3, 4, 5
    chunked_strings = [
        split_sentence_rule1(sentence),
        split_sentence_rule3(sentence),
        split_sentence_rule4(sentence),
        split_sentence_rule5(sentence),
    ]
    merged_chunked_strings = merge_chunked_strings(chunked_strings)

    # Rule 2
    return split_sentence_rule2_chunked_input(merged_chunked_strings)


def extend_short_chunks(chunked_sentence, min_chunk):
    """
    Extend chunks in a text such that each chunk has at least a minimum number of words.

    Args:
    chunked_sentence (str): The text with chunks separated by slashes.
    min_chunk (int): The minimum number of words each chunk should contain.

    Returns:
    str: The text with chunks merged to meet the minimum word count.

    Example:
    >>> extend_short_chunks("So / in my talk today, / I / want / to / share / with you some insights / I've obtained.", 2)
    "So in my talk today, / I want / to share / with you some insights / I've obtained."
    >>> extend_short_chunks("So / in my talk today, / I / want / to / share / with you some insights / I've obtained.", 3)
    "So in my talk today, / I want to / share with you some insights / I've obtained."
    """
    chunks = chunked_sentence.split(" / ")
    merged_chunks = []
    current_chunk = []

    for chunk in chunks:
        current_chunk.append(chunk)
        if sum(len(c.split()) for c in current_chunk) < min_chunk:
            continue
        merged_chunks.append(" ".join(current_chunk))
        current_chunk = []

    # Add any remaining words as a chunk
    if current_chunk:
        merged_chunks.append(" ".join(current_chunk))

    return " / ".join(merged_chunks)


def calculate_f1_precision_recall(predicted, correct):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Loop through each sentence and its predicted chunks
    for pred_chunks, corr_chunks in zip(predicted, correct):
        # Create a set of positions where the correct chunks end
        correct_positions = set()
        position = 0
        for chunk in corr_chunks[:-1]:
            position += len(chunk.split())  # Count the words in the chunk
            correct_positions.add(position)

        # Create a set of positions where the predicted chunks end
        predicted_positions = set()
        position = 0
        for chunk in pred_chunks[:-1]:
            position += len(chunk.split())  # Count the words in the chunk
            predicted_positions.add(position)

        # Calculate TP, FP, FN
        for pos in predicted_positions:
            if pos in correct_positions:
                true_positives += 1
            else:
                false_positives += 1

        for pos in correct_positions:
            if pos not in predicted_positions:
                false_negatives += 1

    # Calculating Precision, Recall, and F1 Score
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return f1, precision, recall


def main():
    parser = argparse.ArgumentParser(
        description="Split a sentence into multiple sentences."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Chunk input sentences in interactive mode.",
    )
    mode_group.add_argument(
        "--sentence-files",
        "-f",
        nargs="+",
        help="List of files containing sentences to be chunked.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-files",
        "-o",
        nargs="+",
        help="List of files to write the chunked sentences to.",
    )
    output_group.add_argument(
        "--output-dir",
        "-d",
        help="Directory to write the chunked sentences to. \
            The output files will be named after the input files.",
    )
    parser.add_argument(
        "--min-chunk",
        "-m",
        type=int,
        default=3,
        help="Minimum number of words in each chunk.",
    )
    parser.add_argument(
        "--do-eval",
        "-e",
        action="store_true",
        help="Evaluate the chunker on the test set.",
    )
    if parser.parse_known_args()[0].do_eval:
        parser.add_argument(
            "--ref-files",
            "-r",
            nargs="+",
            help="List of files containing manually split sentences.",
        )
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print the chunked sentences to the console.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bar.",
    )

    args = parser.parse_args()

    if args.interactive:
        print("Interactive mode with min_chunk = {}".format(args.min_chunk))
        while True:
            sentence = input("Enter a sentence to split: ")
            print(
                extend_short_chunks(
                    split_sentence_by_five_rules(sentence), args.min_chunk
                )
            )
    elif args.sentence_files:
        if args.output_files:
            assert len(args.output_files) == len(
                args.sentence_files
            ), "The number of output files must match the number of input files."

        if args.do_eval:
            assert len(args.ref_files) == len(
                args.sentence_files
            ), "The number of reference files must match the number of input files."

            reference_chunks = []
            ref_curr_sentence = []
            for refentence_file in args.ref_files:
                with open(refentence_file, "r") as f:
                    for line in f:
                        stripped_line = line.strip()
                        if not stripped_line:
                            if ref_curr_sentence:
                                # Add the current sentence to the formatted chunk
                                reference_chunks.append(ref_curr_sentence)
                                ref_curr_sentence = []
                        else:
                            # Add only the text part of the chunk,
                            # excluding any trailing characters like '/4'
                            ref_curr_sentence.append(stripped_line.split(" /")[0])
                if ref_curr_sentence:
                    reference_chunks.append(ref_curr_sentence)
                    ref_curr_sentence = []

        predicted_chunks = []
        for i, sentence_file in enumerate(args.sentence_files):
            with open(sentence_file, "r") as f:
                sentences = f.readlines()
            chunked_sentences = []
            for sentence in tqdm(sentences, disable=args.disable_tqdm):
                sentence = re.sub(r"\s+", " ", sentence).rstrip()
                chunked_sentence = extend_short_chunks(
                    split_sentence_by_five_rules(sentence), args.min_chunk
                )
                chunked_sentences.append(chunked_sentence + "\n")
                if args.print:
                    print(chunked_sentence)
                predicted_chunks.append(chunked_sentence.split(" / "))

            if args.output_files:
                os.makedirs(os.path.dirname(args.output_files[i]), exist_ok=True)
                with open(args.output_files[i], "w") as f:
                    f.writelines(chunked_sentences)
            elif args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(
                    os.path.join(args.output_dir, os.path.basename(sentence_file)), "w"
                ) as f:
                    f.writelines(chunked_sentences)

        if args.do_eval:
            assert len(predicted_chunks) == len(
                reference_chunks
            ), "The number of predicted chunks must match the number of reference chunks."
            f1, precision, recall = calculate_f1_precision_recall(
                predicted_chunks, reference_chunks
            )
            print(f"F1 = {f1:.3f}, Precision = {precision:.3f}, Recall = {recall:.3f}")


if __name__ == "__main__":
    main()
