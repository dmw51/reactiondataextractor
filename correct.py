# -*- coding: utf-8 -*-

"""
This file contains an auto-correction module. It is used to fix small mistakes created during the OCR process.

"""

import re
#  Conditions


class Correct:
    co_reactant_units = ['equiv', 'equivs', 'equivalent', 'equivalents']
    catalyst_units = ['mol%', 'mol']  # if space before %
    confidence = 0.5

    def __init__(self,text, len_n_grams=2, confidence=None, correction_candidates=None):
        self.text = text
        self.len_n_grams = len_n_grams
        self.correction_candidates = (correction_candidates if correction_candidates
                                                           else Correct.co_reactant_units + Correct.catalyst_units)
        self._candidate_n_grams = {word:self._generate_n_grams(word) for word in self.correction_candidates}
        self.confidence = confidence if confidence else Correct.confidence

    @property
    def candidate_n_grams(self):
        return self._candidate_n_grams

    def correct_text(self):
        words = self.text.split(' ')
        new_text = []
        for word in words:
            word= re.sub(r'[(){}<>]', '', word)
            correction_candidates = sorted([(candidate, self.get_comparison_confidence(word, candidate))
                                      for candidate in self.correction_candidates], key = lambda x: x[1], reverse=True)
            print(f'correction_candidates: {correction_candidates}')
            if correction_candidates[0][1] >= self.confidence:
                new_text.append(correction_candidates[0][0])
            else:
                new_text.append(word)
        print(f'new text: {new_text}')
        return ' '.join(new_text)

    def _generate_n_grams(self, text):
        return [text[i:i+self.len_n_grams] for i in range(0, len(text)-self.len_n_grams+1)]

    def get_comparison_confidence(self, word, other):
        word_n_grams = self._generate_n_grams(word)
        expected_n_grams = self.candidate_n_grams[other]  #  Can expand this later
        print(f'expected n_grams : {expected_n_grams}')
        print(f'word n_grams: {word_n_grams}')
        intersection = [n_gram for n_gram in word_n_grams if n_gram in expected_n_grams]  #  Can be done using sets?
        print(f'intersection: {intersection}')
        return len(intersection)/max(len(word_n_grams), len(expected_n_grams))   #  Return confidence






