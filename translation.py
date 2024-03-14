import pandas as pd
import re
import nltk
from nltk.translate import AlignedSent, Alignment, IBMModel3

import preprocessing, modifiers

def train_translation_model(source_words, target_words, classes):
    bitext = [AlignedSent(source.split(), target.split()) for source, target in zip(source_words, target_words)]
    bitext.append(AlignedSent(['kolarvu', 'torelwe'],['I','eat','bread'],Alignment.fromstring('0-2 1-0 1-1')))
    ibm_model = IBMModel3(bitext, 10)
    return ibm_model

def clean_input(sentence):
    sentence = sentence.strip()
    sentence = sentence.lower()
    return preprocessing.puflantu_tokenizer(sentence)

def translate_input(ibm_model, source_text):
    cleaned_text = clean_input(source_text)
    source_words = [t.text for t in cleaned_text if not t.is_space]
    translated_words = []
    for source_word in source_words:
        max_prob = 0.0
        translated_word = None
        for target_word in ibm_model.translation_table[source_word]:
            prob = ibm_model.translation_table[source_word][target_word]
            if prob > max_prob:
                max_prob = prob
                translated_word = target_word
        if translated_word is not None:
            translated_words.append(translated_word)
    translated_text = ' '.join(translated_words)
    print(f"\"{source_text}\" in Puflantu is \"{translated_text}\" in English.")

df = pd.read_csv("tiny_dict.csv")
english_words = df['English'].to_list()
puflantu_words = df['Puflantu'].to_list()
word_classes = df['Class'].to_list()

translation_model = train_translation_model(puflantu_words, english_words, word_classes)

translate_input(translation_model, "kolarvu wlwmoc.")

