# pylint: disable=missing-module-docstring
# pylint: disable=too-few-public-methods
# pylint: disable=line-too-long

from bisect import bisect_left
from itertools import takewhile
from dataclasses import dataclass
from typing import Iterator, Iterable, Tuple, List
from collections import Counter
from .document import Document
from .corpus import Corpus
from .analyzer import Analyzer


class SuffixArray:
    """
    A simple suffix array implementation. Allows us to conduct efficient substring searches.
    The prefix of a suffix is an infix!

    In a serious application we'd make use of least common prefixes (LCPs), pay more attention
    to memory usage, and add more lookup/evaluation features.
    """

    @dataclass
    class Options:
        """
        Query-time options. Controls lookup behavior.
        """
        hit_count: int = 10  # The maximum number of results to return to the client.

    @dataclass
    class Result:
        """
        An individual lookup result, as reported back to the client.
        """
        document: Document  # The document with the matching content.
        score: int          # The number of times the query appears in the matching content.

    def __init__(self, corpus: Corpus, fields: Iterable[str], analyzer: Analyzer):
        self._corpus = corpus
        self._analyzer = analyzer
        self._haystack: List[Tuple[int, str]] = []  # The (<document identifier>, <searchable content>) pairs.
        self._suffixes: List[Tuple[int, int]] = []  # The sorted (<haystack index>, <start offset>) pairs.
        self._build_suffix_array(fields)  # Construct the haystack and the suffix array itself.

    def _build_suffix_array(self, fields: Iterable[str]) -> None:
        """
        Builds a simple suffix array from the set of named fields in the document collection.
        The suffix array allows us to search across all named fields in one go.
        """
        
        # delimiter to separate content for different docs and fields
        delimiter = " \0 "

        for doc in self._corpus:
            text = ""

            for field in fields:
                # get content from field, and normalize through analyzer
                text += self._analyzer.join(str(doc.get_field(field, " ")))
                text += delimiter # separating fields

            # storing doc_id and its text
            self._haystack.append((doc.get_document_id(), text))

        # index for searching
        doc_index = 0

        # goes through the haystack
        for _, content in self._haystack:
            # look at each character position in the document's content
            for offset in range(len(content)):
            # checks if offset is at the start (0) or at the start of a word's boundary
            # example: in the text "Hello world" only pos 0 and 6 is valid
                if offset == 0 or content[offset - 1] in (" ", "\0"):
                    # store pointer to this suffix: (doc_index, character_pos)
                    self._suffixes.append((doc_index, offset))
            doc_index += 1 # next posision

        # sort suffixes alphabetically
        self._suffixes.sort(
            key=lambda 
            pair: self._haystack[pair[0]][1][pair[1]:]
            )
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def _get_suffix(self, pair: Tuple[int, int]) -> str:
        """
        Produces the suffix/substring from the normalized document buffer for the given (index, offset) pair.
        """
        index, offset = pair
        return self._haystack[index][1][offset:]  # Slicing implies copying. This should be possible to avoid.

    def evaluate(self, query: str, options: Options | None = None) -> Iterator[Result]:
        """
        Evaluates the given query, doing a "phrase prefix search".  E.g., for a supplied query phrase like
        "to the be", we return documents that contain phrases like "to the bearnaise", "to the best",
        "to the behemoth", and so on. I.e., we require that the query phrase starts on a token boundary in the
        document, but it doesn't necessarily have to end on one.

        The matching documents are ranked according to how many times the query substring occurs in the document,
        and only the "best" matches are yielded back to the client. Ties are resolved arbitrarily.
        """

        # normalizing query using analyzer 
        query_norm = self._analyzer.join(query)

        # handles emty query cases
        if not query_norm:
            return iter([])

        # help method checking if suffix starts with our query
        def matching(pair: Tuple[int, int]) -> bool:
            return self._get_suffix(pair).startswith(query_norm)
        
        # binary search to quickly find starting pos with our query
        start = bisect_left(
            self._suffixes,     # sorted list for binary search
            query_norm,         # what we are looking for
            key=lambda pair: self._get_suffix(pair) # how its extracted
        )

        # collect matching suffixes using takewhile
        # NOTE TO SELF: all matches will be consecutive due to suffixes being sorted, 
        # therefore the takewhile will stop when a suffix doesn't match
        matching_suffixes = takewhile(
            matching, self._suffixes[start:] # uses starting position for the binary search ^
        )

        # counter to check occurences of query in a doc
        counts = Counter()

        # checks for and updates occurences
        for hay_idx, _ in matching_suffixes:
            # gets doc_id for current suffix
            doc_id = self._haystack[hay_idx][0]
            counts.update([doc_id]) # incremets the counter for that doc

        # returns the most common documents sorted by highest counts, 
        # and limits result based on hit_count from given options
        for doc_id, score in counts.most_common(options.hit_count):
            yield SuffixArray.Result(self._corpus.get_document(doc_id), score)
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")
