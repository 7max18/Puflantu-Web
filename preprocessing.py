import re
import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab

df = pd.read_csv("puflantu_eng_dict.csv")
vocab = Vocab(strings=df['Puflantu'].to_list())
special_cases = dict()
prefix_re = re.compile(r'''^"''')
suffix_re = re.compile(r'''[,."]$''')
infix_re = re.compile(r'''-''')

puflantu_tokenizer = Tokenizer(vocab, rules=special_cases,
                               prefix_search=prefix_re.search,
                               suffix_search=suffix_re.search,
                               infix_finditer=infix_re.finditer)