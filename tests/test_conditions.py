import pytest
from pprint import pprint

from chemdataextractor.doc.text import Sentence, Span
from conditions import ConditionParser

SENTENCES = [Sentence(text='LiOH (3 equiv.)'), Sentence(text='KMnO3 (1.5 equivalents'), Sentence(text='MnO2 (0.5 mol%'),
             Sentence(text='MeOH, BuOH, 2h'), Sentence(text='60oC, 2 bar')]


def test_identify_chemicals():
    parser = ConditionParser(SENTENCES)
    species = [parser._identify_species(sentence) for sentence in parser.sentences]
    print(f'all species:  {species}')
    print('---------------')
    assert species == [[Span('LiOH',0,4)],
                       [Span('KMnO3',0,5)],
                       [Span('MnO2',0,4)],
                       [Span('MeOH',0,4), Span('BuOH',6,10)],
                       []]

def test_parse_co_reactants():
    parser = ConditionParser(SENTENCES)
    co_reactants = [parser._parse_coreactants(sentence) for sentence in parser.sentences]
    print(f'co_reactants: {co_reactants}')
    print('---------------')
    assert co_reactants == [[{'Species': Span('LiOH',0,4), 'Value': 3, 'Units': 'equiv.'}],
                            [{'Species': Span('KMnO3',0,5), 'Value': 1.5, 'Units': 'equivalents'}],
                            [],[],[]]


def test_parse_catalysis():
    parser = ConditionParser(SENTENCES)
    catalysts = [parser._parse_catalysis(sentence) for sentence in parser.sentences]
    print(f'catalysts:{catalysts}')
    print('---------------')
    assert catalysts == [[], [],
                        [{'Species': Span('MnO2',0,4), 'Value': 0.5, 'Units': 'mol%'}],
                         [],[]]


def test_parse_other_species():
    parser = ConditionParser(SENTENCES)
    other_species = [parser._parse_other_species(sentence) for sentence in parser.sentences]
    print(f'other species: {other_species}')
    print('---------------')
    assert other_species == [[],[],[],
                             [Span('MeOH',0,4), Span('BuOH',6,10)],
                             []]



def test_other_conditions():
    parser = ConditionParser(SENTENCES)
    other_conditions = [parser._parse_other_conditions(sentence) for sentence in parser.sentences]
    print(f'other_conditions: {other_conditions}')
    print('---------------')
    assert other_conditions == [{}, {}, {}, {'time': {'Value': 2.0, 'Units': 'h'}},
                                            {'temperature': {'Value': 60.0, 'Units': 'oC'},
                                             'pressure': {'Value': 2.0, 'Units': 'bar'}}]


def test_parse_conditions():
    parser = ConditionParser(SENTENCES)
    conditions_dct = parser.parse_conditions()
    pprint(conditions_dct)
    assert conditions_dct == {'catalysts': [{'Species': Span('MnO2',0,4), 'Value': 0.5, 'Units': 'mol%'}],
                              'co-reactants': [{'Species': Span('LiOH',0,4), 'Value': 3.0, 'Units': 'equiv.'},
                                               {'Species': Span('KMnO3',0,5), 'Value': 1.5, 'Units': 'equivalents'}],
                              'other species': [Span('MeOH',0,4), Span('BuOH',6,10)] ,
                              'time': {'Value': 2.0, 'Units': 'h'},
                              'temperature': {'Value': 60.0, 'Units': 'oC'},
                              'pressure': {'Value': 2.0, 'Units': 'bar'}}


if __name__ == '__main__':
    test_identify_chemicals()
    test_parse_co_reactants()
    test_parse_catalysis()
    test_parse_other_species()
    test_other_conditions()
    test_parse_conditions()
