import spacy
import collections
import itertools
import numpy
import pickle
import json
import os
import sqlite3
from spacy.tokens import Doc
from spacy.lang.en.stop_words import STOP_WORDS

spacy_model = spacy.load('en')


class PMIModel():
    model_obj_filename = 'pmi_class_obj.pkl'

    def __init__(self, save_dirpath, min_word_count=1, min_word_pair_count=1,
                 lemmatize=False, filter_stops=False, filtered_pos=[], min_pair_distance=1):
        self.save_dirpath = save_dirpath
        if not os.path.isdir(self.save_dirpath):
            os.mkdir(self.save_dirpath)
        self.min_word_count = min_word_count
        self.min_word_pair_count = min_word_pair_count
        self.lemmatize = lemmatize
        self.filter_stops = filter_stops
        self.filtered_pos = filtered_pos
        self.min_pair_distance = min_pair_distance
        self.vocab = None
        self.word_counts_filepath = os.path.join(self.save_dirpath, 'word_counts.json')
        self.word_pair_counts_filepath = os.path.join(self.save_dirpath, 'word_pair_counts.db')
        self.model_obj_filepath = os.path.join(self.save_dirpath, self.model_obj_filename)

    def initialize_with_counts(self, texts, tokenize_on_space=False):
        self.make_word_counts_file(texts, tokenize_on_space)
        self.make_word_pair_counts_file(texts, tokenize_on_space)

        with open(self.model_obj_filepath, 'wb') as f:
            pickle.dump(self, f)

        print("Saved model object to", self.model_obj_filepath)

    def extract_words(self, text, tokenize_on_space=False, unique_only=True, in_vocab_only=True):
        '''Tokenize the given text into words.
        Remove stop words if self.filter_stops=True.
        Remove all words with POS tags in self.filtered_pos.
        Convert each word to its lemma if self.lemmatize=True.
        If self.vocab is defined, return only the words included in this set.
        If tokenize_on_space=True, assume words in text are already tokenized and tokens are joined by a single white space.
        If unique_only=True, only return the unique words in the text (remove duplicates; only the first index of the word will be returned).
        If in_vocab_only=True, only return the words that appear in self.vocab (does not apply if self.vocab is None).
        Also track the indices of the returned tokens in the original text.
        This is used when min_pair_distance > 1 in order to know the distance between words in the original text
        (and thus to know which words to pair together).'''

        try:
            if tokenize_on_space:
                # Must initialize tagger separately when spacy Doc is given pretokenized input
                spacy_tagger = spacy_model.get_pipe('tagger')
                words, word_idxs = zip(*[(word, word_idx) for word_idx, word
                                         in enumerate(spacy_tagger(Doc(spacy_model.vocab,
                                                                       words=[word for word in text.split(" ") if word.strip()])))
                                         ])
            else:
                words, word_idxs = zip(*[(word, word_idx) for word_idx, word
                                         in enumerate(spacy_model(text)) if not word.is_space])

            if self.filter_stops:  # spacy currently has a bug where attribute word.is_stop doesn't recognized capitalized stop words, so lookup from list
                words, word_idxs = zip(*[(word, word_idx) for word, word_idx in zip(words, word_idxs) if word.lower_.strip()
                                         not in STOP_WORDS])
            if self.filtered_pos:
                words, word_idxs = zip(*[(word, word_idx) for word, word_idx in zip(words, word_idxs)
                                         if word.pos_ not in self.filtered_pos])

            if self.lemmatize:
                words = [word.lemma_.strip() for word in words]
            else:
                words = [word.lower_.strip() for word in words]

            if in_vocab_only and self.vocab:
                words, word_idxs = zip(*[(word, word_idx) for word, word_idx in zip(words, word_idxs)
                                         if word in self.vocab])

            if unique_only:
                unique_words_with_idxs = collections.OrderedDict()
                for word, word_idx in zip(words, word_idxs):
                    if word not in unique_words_with_idxs:
                        unique_words_with_idxs[word] = word_idx
                words, word_idxs = zip(*unique_words_with_idxs.items())

        except ValueError:  # No words that meet criteria
            words, word_idxs = (), ()

        return words, word_idxs

    def extract_word_pairs_in_text(self, text, tokenize_on_space=False):
        # Only form pairs between words that are at least N=min_distance_between_words away from each other in the text
        # For example, if min_distance_between_words = 2, a pair will not be formed between a word and the immediate next word,
        # but it will be formed between that word and the word after the next word

        # Each unique word pair will only have one count for a given text, regardless of how many times it appears.
        # Sort words in pair alphabetically to have consistent ordering when looking up a word pair;
        # The co-occurrence count of a pair does not consider the order in which
        # the words appear in the text.
        # import pdb
        # pdb.set_trace()
        word_pairs = []
        words, orig_word_idxs = self.extract_words(text, tokenize_on_space)
        for cur_word_idx, (word1, orig_word1_idx) in enumerate(zip(words, orig_word_idxs)):
            for word2, orig_word2_idx in zip(words[cur_word_idx + 1:], orig_word_idxs[cur_word_idx + 1:]):
                if orig_word2_idx >= orig_word1_idx + self.min_pair_distance:
                    word_pair = tuple(sorted((word1, word2)))
                    word_pairs.append(word_pair)
        return word_pairs

    def make_word_counts_file(self, texts, tokenize_on_space=False):
        '''Get the individual word frequencies of all words appearing
        at least N=self.min_word_count texts.'''
#         pdb.set_trace()
        word_counts = collections.Counter()
        for text_idx, text in enumerate(texts):
            words, _ = self.extract_words(text, tokenize_on_space)
            words = set(words)
            # In a single text, each word will receive a count of one regardless of how many times it appears in that text.
            # Thus the parameter self.min_word_count refers to a minimum number of
            # texts the word must appear in, not just raw frequency
            word_counts.update(words)
            if text_idx and not text_idx % 2500:
                print("Gathered word counts for", text_idx, "texts...")

        word_counts = {word: count for word, count
                       in word_counts.items()
                       if count >= self.min_word_count}

        self.vocab = set(word_counts.keys())

        print("Collected counts for", len(word_counts),
              "words appearing at least", self.min_word_count, "texts")

        with open(self.word_counts_filepath, 'w') as f:
            json.dump(word_counts, f)

        print("Saved word counts file to", self.word_counts_filepath)
        # return word_counts

    def make_word_pair_counts_file(self, texts, tokenize_on_space=False, db_upload_freq=50000):
        '''Extract all pairs of words that appear in the same text, and save their co-occurence frequencies.
        Only pairs where both words appear in self.word_counts will be considered.
        Ultimately filter pairs that occur in fewer in than N=self.min_word_pair_count texts.'''
        # import pdb
        # pdb.set_trace()
        assert self.vocab is not None

        word_pair_counts_db = sqlite3.connect(self.word_pair_counts_filepath)
        db_cursor = word_pair_counts_db.cursor()

        # need to create bigram counts db if it hasn't been created
        db_cursor.execute("CREATE TABLE IF NOT EXISTS word_pair_counts(\
                          word1 TEXT,\
                          word2 TEXT,\
                          count INTEGER DEFAULT 0,\
                          PRIMARY KEY (word1, word2))")

        word_pair_counts_db.commit()

        # Keep a cache of word pair counts that will be written to db after every
        # N=db_upload_freq texts
        word_pair_counts_cache = {}

        for text_idx, text in enumerate(texts):
            word_pairs = self.extract_word_pairs_in_text(text, tokenize_on_space)
            for word_pair in word_pairs:
                if word_pair not in word_pair_counts_cache:
                    word_pair_counts_cache[word_pair] = 1
                else:
                    word_pair_counts_cache[word_pair] += 1
            if text_idx and (text_idx % 2500 == 0 or text_idx == len(texts) - 1):
                print("Gathered word pair counts for",
                      text_idx, "texts...")
            if text_idx and (text_idx % db_upload_freq == 0 or text_idx == len(texts) - 1):
                # insert words if they don't already exist
                print("Inserting new word pairs into db...")
                db_cursor.executemany("INSERT OR IGNORE INTO word_pair_counts(word1, word2)\
                                      VALUES (?, ?)",
                                      [(word1, word2) for word1, word2 in word_pair_counts_cache])

                print("Updating word pair counts in db...")
                db_cursor.executemany("UPDATE word_pair_counts\
                                      SET count = (count + ?)\
                                      WHERE word1 = ? AND word2 = ?",
                                      [(count, word1, word2) for (word1, word2), count in word_pair_counts_cache.items()])

                print("Finalizing db update...")
                word_pair_counts_db.commit()
                word_pair_counts_cache = {}  # Clear out cache after uploading to db

        # Filter word pairs in db to save only ones that have count >= self.min_word_pair_count
        db_cursor.execute("DELETE from word_pair_counts where count < ?",
                          (self.min_word_pair_count,))
        word_pair_counts_db.commit()

        db_cursor.execute("SELECT COUNT(*) FROM word_pair_counts")
        n_word_pairs = db_cursor.fetchone()[0]
        print("Collected counts for", n_word_pairs,
              "word pairs appearing at least", self.min_word_pair_count, "texts")

        db_cursor.close()
        word_pair_counts_db.close()
        print("Saved word pair counts file to", self.word_pair_counts_filepath)

    def compute_pmi_scores(self, min_word_count=0, min_word_pair_count=0, min_percentile=0.0):
        '''Compute pmi scores for all word pairs in self.word_pair_counts along with their percentiles'''
        # import pdb
        # pdb.set_trace()

        with open(self.word_counts_filepath) as f:
            word_counts = json.load(f)

        word_pair_counts_db = sqlite3.connect(self.word_pair_counts_filepath)
        db_cursor = word_pair_counts_db.cursor()

        db_cursor.execute("SELECT * FROM word_pair_counts")

        self.pmi_scores = {}

        for idx, (word1, word2, word_pair_count) in enumerate(db_cursor):
            word1_count = word_counts[word1]
            word2_count = word_counts[word2]
            if word_pair_count >= min_word_pair_count and word1_count >= min_word_count and word2_count >= min_word_count:
                score = self.compute_pmi_score_from_counts(word1_count, word2_count,
                                                           word_pair_count)
                self.pmi_scores[(word1, word2)] = score
            # if idx == 1000000:
            #     break

        if min_percentile:
            score_at_percentile = self.get_pmi_score_at_percentile(min_percentile)

        # Filter all pairs with scores below percentile threshold
        self.pmi_scores = {word_pair: score for word_pair, score in self.pmi_scores.items()
                           if not min_percentile or score >= score_at_percentile}

        self.min_pmi_score = min(self.pmi_scores.values())

        db_cursor.close()
        word_pair_counts_db.close()

        print("Computed PMI scores for", len(self.pmi_scores), "word pairs")

        # return self.pmi_scores

    def compute_pmi_score_from_counts(self, word1_count, word2_count, word_pair_count):
        '''Return the Pointwise Mutual Information (PMI) score of two words based on the given counts'''

        # Return PMI as log score (add tiny value to avoid taking log of zero)
        # score = word_pair_count / (word1_count * word2_count)
        score = numpy.log(word_pair_count) - numpy.log(word1_count) - numpy.log(word2_count)

        return score

    def look_up_pmi_score(self, word1, word2):
        '''Return the pmi score for the given pair of words'''

        # Ensure word1 and word2 assignment is based on alphabetic ordering
        word1, word2 = sorted([word1, word2])
        # Return tiny non-zero value if word pair not in scores
        if (word1, word2) not in self.pmi_scores:
            score = numpy.log(1e-10)
        else:
            score = self.pmi_scores[(word1, word2)]
        return score

    def get_pmi_score_at_percentile(self, percentile):
        '''Return the pmi score that appears at the given percentile in self.pmi_scores'''

        score_at_percentile = numpy.percentile(numpy.fromiter(self.pmi_scores.values(),
                                                              dtype=float),
                                               percentile * 100)
        return score_at_percentile

    # def get_word_pairs_above_percentile(self, min_percentile):
    #     '''Return all word pairs whose PMI score is greater than or equal to the score at the given percentile'''

    #     score_at_percentile = self.get_pmi_score_at_percentile(min_percentile)
    #     word_pairs_above_percentile = [(word1, word2) for (word1, word2), score
    #                                    in self.pmi_scores.items() if score >= score_at_percentile]
    #     return word_pairs_above_percentile

    def get_mean_pmi_score_for_seq_pair(self, seq1, seq2):
        '''Calculate PMI scores of all word pairs between seq1 and seq2.
        Return the mean score.'''

        seq1_words = self.extract_words(seq1)
        seq2_words = self.extract_words(seq2)

        word_pair_scores = [self.look_up_pmi_score(word1, word2)
                            for word1 in seq1_words
                            for word2 in seq2_words]
        if not word_pair_scores:  # If no word pairs in vocab found, return 0 as mean score
            mean_score = numpy.log(1e-10)
        else:
            mean_score = numpy.mean(word_pair_scores)
        return mean_score

    # def get_n_candidate_pairs_in_text(self, text):

    #     words = set(self.extract_words(text))
    #     n_candidate_pairs = len(list(itertools.combinations(words, 2)))
    #     return n_candidate_pairs

    def get_pmi_scores_for_pairs(self, word_pairs):

        word_pair_scores = sorted([(word_pair, self.pmi_scores[word_pair] if word_pair in self.pmi_scores else self.min_pmi_score)
                                   for word_pair in word_pairs], key=lambda word_pair_score:
                                  word_pair_score[-1], reverse=True)
        # word_pair_scores = sorted([(word_pair, self.pmi_scores[word_pair])
        #                            for word_pair in word_pairs if word_pair in self.pmi_scores],
        #                           key=lambda word_pair_score: word_pair_score[-1], reverse=True)
        return word_pair_scores

    # def get_sum_of_pmi_scores_for_pairs(self, word_pair_scores):
    #     if not word_pair_scores:
    #         sum_pmi_score = numpy.log(1e-10)
    #     else:
    #         # - numpy.log(len(word_pair_scores))
    #         sum_pmi_score = logsumexp([score for word_pair, score in word_pair_scores])
    #         #sum_pmi_score = numpy.median([score for word_pair, score in word_pair_scores])

    #     return sum_pmi_score

    def __getstate__(self):
        # Don't save pmi scores, they'll be saved separately as json
        attrs = self.__dict__.copy()
        # if 'word_counts' in attrs:
        #     del attrs['word_counts']
        # if 'word_pair_counts' in attrs:
        #     del attrs['word_pair_counts']
        if 'pmi_scores' in attrs:
            del attrs['pmi_scores']
        return attrs

    @classmethod
    def load(cls, dirpath):
        # import pdb
        # pdb.set_trace()
        model_obj_filepath = os.path.join(dirpath, cls.model_obj_filename)
        with open(model_obj_filepath, 'rb') as f:
            model = pickle.load(f)
        print("Loaded model object from", os.path.join(model_obj_filepath))
        return model
