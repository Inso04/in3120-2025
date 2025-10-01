# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

from dataclasses import dataclass
from typing import Iterator, Any, List
from .analyzer import Analyzer
from .trie import Trie


class StringFinder:
    """
    Given a trie encoding a dictionary of strings, efficiently finds the subset of strings in the dictionary
    that are also present in a given text buffer. I.e., in a sense computes the "intersection" or "overlap"
    between the dictionary and the text buffer.

    Uses a trie-walk algorithm similar to the Aho-Corasick algorithm with some simplifications (we ignore the
    part about failure transitions) and some minor NLP extensions. The running time of this algorithm is in
    practice virtually insensitive to the size of the dictionary, and linear in the length of the buffer we
    are searching in.

    The analyzer we use when scanning the input buffer is assumed to be the same as the one that was used
    when adding strings to the trie.
    """

    @dataclass
    class State:
        """
        A currently explored state, as the scan proceeds.
        """
        node: Trie  # The current position in the trie, after having consumed zero or more characters.
        begin: int  # The index into the original buffer where the state was "born".
        match: str  # The symbols consumed so far to get to the current state.

    @dataclass
    class Result:
        """
        An individual result of the scan, as reported back to the client.
        """
        match: str        # The matching dictionary entry.
        meta: None | Any  # Optional mata data associated with the match, if present in the dictionary.
        surface: str      # The part of the input buffer that triggered the match, space-normalized.
        begin: int        # The index into the original buffer where the surface form starts.
        end: int          # The index into the original buffer where the surface form ends.

    def __init__(self, trie: Trie, analyzer: Analyzer):
        self._trie = trie          # The set of strings we want to detect in the scanned buffer.
        self._analyzer = analyzer  # The same that was used when the trie was built.

    def scan(self, buffer: str) -> Iterator[Result]:
        """
        Scans the given buffer once and finds all dictionary entries in the trie that are also present in the
        buffer. We only consider matches that begin and end on token boundaries.

        In a serious application we'd add more lookup/evaluation features, e.g., support for prefix matching,
        support for leftmost-longest matching (instead of reporting all matches), and more.
        """

        # tokenize and normalize, returns (term, span)
        terms = self._analyzer.terms(buffer)

        # "bookmark" for potential matches
        # NOTE TO SELF: type hint is not "necessary" but added readability + import 
        active_states: List[self.State] = []

        # process tokens one by one
        # goes through (term, span) -> span = (begin, end)
        for token, (token_begin, token_end) in terms:
            # start new potential match at tokens pos
            new_state = self.State(
                self._trie, # start at root of dict
                token_begin,# remember where this potential match started
                "")         # No char matched yet
            # append to list
            active_states.append(new_state)

            # process each character of current token
            for char in token:
                surviving_states: List[self.State] = []
                
                # goes through active states
                for state in active_states:
                    next_node = state.node.child(char)

                    # checks if current trie node ^accepts char
                    if next_node is not None:
                        # if true
                        state.node = next_node  # move to next trie node
                        state.match += char     # add char to match
                        surviving_states.append(state)# keep state
                    # if not true, state dies (no match)
                # replace old with "surviving" active states
                active_states = surviving_states

            # after process, check if any states complete match
            new_surviving_states: List[self.State] = []

            for state in active_states:
                # checks if state is at final node
                if state.node.is_final():
                    # found match
                    phrase = buffer[state.begin : token_end]
                    clean_phrase = " ".join(phrase.split()) # normalize whitespace
                    
                    # yield match
                    yield self.Result(
                        state.match,    # matched dict entry
                        state.node.get_meta(),  # match's metadata
                        clean_phrase,   # how it appears in original
                        state.begin,    # start pos in orgininal buffer
                        token_end)  # end pos in original buffer
                    
                # check if match can potentially continue
                # if more text exists, next char is a space, and trie can continue with space
                next_node = state.node.child(" ")
                if (token_end < len(buffer) and buffer[token_end] == " " and (next_node is not None)):
                    # can continue to next token
                    state.node = next_node  # advance
                    state.match += " "      # adds space to match
                    new_surviving_states.append(state)  # keep for next iter

            # keep states that can continue
            if token_end < len(buffer) and buffer[token_end] == " ":
                active_states = new_surviving_states
            # if no space, can't continue
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")
