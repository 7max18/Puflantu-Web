import pandas as pd
import re
import nltk
from nltk.translate import AlignedSent, IBMModel1

def train_translation_model(source_sentences, target_sentences):
    aligned_sentences = [AlignedSent(source.split(), target.split()) for source, target in zip(source_sentences, target_sentences)]
    ibm_model = IBMModel1(aligned_sentences, 10)
    return ibm_model

def clean_input(sentence):
    cleaned_sentence = list()
    for word in sentence:
        word = word.strip()
        word = word.lower()
        word = re.sub(r"[^a-zA-Z0-9']+", "", word)
        cleaned_sentence.append(word)
    return cleaned_sentence

def translate_input(ibm_model, source_text):
    cleaned_text = clean_input(source_text.split())
    source_words = cleaned_text
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

df = pd.read_csv("puflantu_eng_dict.csv")
english_words = df['English'].to_list()
puflantu_words = df['Puflantu'].to_list()

translation_model = train_translation_model(puflantu_words, english_words)

translate_input(translation_model, "Torelwe wlwmoc.")

