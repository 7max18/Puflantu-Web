import pandas as pd
import re
import nltk
from nltk.translate import AlignedSent, Alignment, IBMModel3, PhraseTable
from nltk.translate.stack_decoder import StackDecoder
from nltk.translate.phrase_based import phrase_extraction

import preprocessing, modifiers

def train_translation_model(target_words, source_words):
    bitext = [AlignedSent(target.split(), source.split()) for target, source in zip(target_words, source_words)]
    bitext.append(AlignedSent(['I','eat'],['wlwmoc']))
    bitext.append(AlignedSent(['I','eat','bread'],['kolarvu', 'wlwmoc'], Alignment.fromstring('0-1 1-1 2-0')))
    ibm_model = IBMModel3(bitext, 10)

    return ibm_model

def clean_input(sentence):
    sentence = sentence.strip()
    sentence = sentence.lower()
    return preprocessing.puflantu_tokenizer(sentence)

def translate_input(ibm_model:IBMModel3, source_text):
    cleaned_text = clean_input(source_text)
    source_words = [t.text for t in cleaned_text if t.text.isalpha()]
    translated_text = ' '.join(stack_decoder.translate(source_words))
    print(f"\"{source_text}\" in Puflantu is \"{translated_text}\" in English.")



df = pd.read_csv("puflantu_eng_dict.csv")
english_words = df['English'].to_list()
puflantu_words = df['Puflantu'].to_list()

translation_model = train_translation_model(english_words, puflantu_words)
srctext = "michael assumes that he will stay in the house"
trgtext = "michael geht davon aus , dass er im haus bleibt"
alignment = [(0,0), (1,1), (1,2), (1,3), (2,5), (3,6), (4,9), (5,9), (6,7), (7,7), (8,8)]
print(phrase_extraction(srctext, trgtext, alignment))
# stack_decoder = StackDecoder(translation_model.translation_table, translation_model)
# translate_input(translation_model, "kolarvu wlwmoc.")

