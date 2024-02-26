import yaml
import nltk
from urbans import Translator
with open('dict.yaml', 'r') as f:
    pf_en_dict = yaml.safe_load(f)

pf_grammar = """
                VP -> NP VB
                NP -> NJ
             """

with open('nounjectives.txt', 'r') as f:
    pf_grammar += "NJ -> " + "|".join(f)
with open('verbs.txt', 'r') as f:
    pf_grammar += "VB -> " + "|".join(f)

src_to_tgt_grammar = {"NJ": "NN", "VP -> NP VB" : "VP -> VB NP"}

translator = Translator(src_grammar=src_grammar, 
                        src_to_tgt_grammar=src_to_tgt_grammar, 
                        src_to_tgt_dictionary=pf_en_dict)
print(translator.translate("torewle wlwmoc"))