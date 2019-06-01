import sys
import os
import pandas
import numpy
from xml.etree import cElementTree


class StoryClozeTest():

    def __init__(self, dirpath, test_filename, valid_filename=None):
        self.test_cloze_items = self.load_cloze_items(filepath=os.path.join(dirpath,
                                                                            test_filename))
        if valid_filename:
            self.valid_cloze_items = self.load_cloze_items(filepath=os.path.join(dirpath,
                                                                                 valid_filename))

    def load_cloze_items(self, filepath):
        cloze_items = pandas.read_csv(filepath, sep=',', encoding='utf-8')
        # Ensure readable encoding
        # cloze_items = cloze_items[cloze_items[['InputSentence1', 'InputSentence2', 'InputSentence3',
        #                                        'InputSentence4', 'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2']].apply(
        # lambda sentences: numpy.all([type(sentence) is unicode for sentence in
        # sentences]), axis=1)]
        input_stories = cloze_items[['InputSentence1', 'InputSentence2',
                                     'InputSentence3', 'InputSentence4']].values.tolist()
        # Combine sents for each story into single string
        input_stories = [" ".join(story) for story in input_stories]

        output_choices = pandas.concat([cloze_items[['RandomFifthSentenceQuiz1',
                                                     'RandomFifthSentenceQuiz2']]], axis=1).values.tolist()

        # Subtract 1 from gold ending choice index so indices start at 0
        correct_answers = pandas.concat([cloze_items['AnswerRightEnding']],
                                        axis=1).values.flatten() - 1

        assert len(input_stories) == len(output_choices) == len(correct_answers)
        cloze_items = list(zip(input_stories, output_choices, correct_answers))
        return cloze_items  # Return cloze as list of items, where each item contains the input story, ending choices, and correct answer index

    def evaluate(self, score_fn, partition='test'):
        '''Apply a score function associated with a model to predict answers for the items in the partition (valid or test).
        The prediction function itself will take a story and its corresponding choices as input and return a predicted answer.
        Return all predicted answers as well as their overall accuracy based on the gold answers'''

        # import pdb
        # pdb.set_trace()

        assert partition in ('valid', 'test')
        if partition == 'valid':
            cloze_items = self.valid_cloze_items
        elif partition == 'test':
            cloze_items = self.test_cloze_items

        pred_answers = [numpy.argmax([score_fn(story, choice1), score_fn(story, choice2)])
                        for story, (choice1, choice2), _ in cloze_items]
        correct_answers = [answer for _, _, answer in cloze_items]
        accuracy = numpy.mean(pred_answers == numpy.array(correct_answers))
        return pred_answers, accuracy


class COPA():

    def __init__(self, dirpath, test_filename, valid_filename=None):
        self.test_copa_items = self.load_copa_items(filepath=os.path.join(dirpath,
                                                                          test_filename))
        if valid_filename:
            self.valid_copa_items = self.load_copa_items(filepath=os.path.join(dirpath,
                                                                               valid_filename))

    def load_copa_items(self, filepath):
        xml_tree = cElementTree.parse(filepath)
        corpus = xml_tree.getroot()
        premises = []
        alts = []
        answers = []
        modes = []
        for item in corpus:
            mode = item.attrib["asks-for"]
            modes.append(mode)
            answer = int(item.attrib["most-plausible-alternative"]) - \
                1  # answers are 1 and 2, convert to 0 and 1
            answers.append(answer)
            premise = item.find("p").text
            premises.append(premise)
            alt1 = item.find("a1").text
            alt2 = item.find("a2").text
            alts.append([alt1, alt2])
        answers = numpy.array(answers)
        # Return COPA as list of items, where each item contains the input
        # premise, choice alternatives, correct answer index, and mode (i.e.
        # whether the premise elicits the cause or effect)
        copa_items = list(zip(premises, alts, answers, modes))

        return copa_items

    def evaluate(self, score_fn, partition='test'):

        # import pdb
        # pdb.set_trace()

        assert partition in ('valid', 'test')
        if partition == 'valid':
            copa_items = self.valid_copa_items
        elif partition == 'test':
            copa_items = self.test_copa_items

        correct_answers = []
        pred_answers = []

        for premise, (alt1, alt2), correct_answer, mode in copa_items:
            if mode == 'cause':
                pred_answer = numpy.argmax([score_fn(alt1, premise), score_fn(alt2, premise)])
            elif mode == 'effect':
                pred_answer = numpy.argmax([score_fn(premise, alt1), score_fn(premise, alt2)])
            correct_answers.append(correct_answer)
            pred_answers.append(pred_answer)

        accuracy = numpy.mean(pred_answers == numpy.array(correct_answers))
        return pred_answers, accuracy
