import pandas as pd
import re
import nltk
from nltk.translate import AlignedSent, IBMModel3

import preprocessing, modifiers

def train_translation_model(target_words, source_words):
    bitext = [AlignedSent(target.split(), source.split()) for target, source in zip(target_words, source_words)]
    bitext.append(AlignedSent(['I','eat'],['wlwmoc']))
    bitext.append(AlignedSent(['I','eat','bread'],['kolarvu', 'wlwmoc', 'zo']))
    ibm_model = IBMModel3(bitext, 10)
    print(ibm_model.fertility_table[1]['kolarvu'])
    return ibm_model

def clean_input(sentence):
    sentence = sentence.strip()
    sentence = sentence.lower()
    return preprocessing.puflantu_tokenizer(sentence)

def translate_input(ibm_model:IBMModel3, source_text):
    cleaned_text = clean_input(source_text)
    source_words = [t.text for t in cleaned_text if t.text.isalpha()]
    fertile_words = list()
    for word in source_words:
        fertility_prob = 0.0
        fertility_num = 0
        for num in ibm_model.fertility_table.keys():
            cur_fertility = ibm_model.fertility_table[num][word]
            if cur_fertility > fertility_prob:
                fertility_prob = cur_fertility
                fertility_num = num
        for i in range(fertility_num):
            fertile_words.append(word)
    translated_words = list()
    for source_word in fertile_words:
        translation_prob, translated_word = max([(ibm_model.translation_table[x][source_word], x) for x in ibm_model.translation_table if x not in translated_words])
        if translated_word is not None:
            translated_words.append(translated_word)
    #Distortion table: [trg_pos][src_pos][src_len][trg_len]
    alignments = list()
    for i in range(1, len(source_words)+1):
        max_align_prob = 0.0
        align_pos = -1
        for trg_pos, value in ibm_model.distortion_table[i].items():
            cur_max = max([x[len(source_text)] for x in value.values()])
            if cur_max > max_align_prob:
                max_align_prob = cur_max
                align_pos = trg_pos
        alignments.append(align_pos)
    print(alignments)
    translated_text = ' '.join(translated_words)
    print(f"\"{source_text}\" in Puflantu is \"{translated_text}\" in English.")

df = pd.read_csv("puflantu_eng_dict.csv")
english_words = df['English'].to_list()
puflantu_words = df['Puflantu'].to_list()

translation_model = train_translation_model(english_words, puflantu_words)

translate_input(translation_model, "kolarvu wlwmoc.")

