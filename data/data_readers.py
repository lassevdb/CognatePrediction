"""
Decoder-only Cognate Prediction
Lasse van den Berg, Adnan Bseisu
CPSC 477, Spring 2024

This file contains necessary data preprocessing functions.
It accesses the CogNet databse and extracts cognate pairs in specified languages.
It also transliterates words into IPA format using a lookup table.
"""



import pandas as pd
from typing import List, Dict

def cognet_reader(lang_1, lang_2):

    file_path = 'data/cognate_data/CogNet-v1.0.tsv'
    data = pd.read_csv(file_path, sep='\t')

    cognates = []
    for i in range(len(data)):
        if data["lang 1"][i] == lang_1 and data["lang 2"][i] == lang_2:
            if type(data["word 1"][i]) == str and type(data["word 2"][i]) == str:
                cognates.append([data["word 1"][i], data["word 2"][i]])
        elif data["lang 1"][i] == lang_2 and data["lang 2"][i] == lang_1:
            if type(data["word 1"][i]) == str and type(data["word 2"][i]) == str:
                cognates.append([data["word 1"][i], data["word 2"][i]])
    print(f"Found {len(cognates)} cognate pairs!")
    return cognates

def ipa_lookup_reader(cleanup=True):
    def load_ipa_data(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            ipa_data = {}
            for line in file:
                parts = line.strip().split('\t')
                if "," in parts[1]:
                    subparts = parts[1].strip().split(",")
                    parts = parts[0], subparts[0]
                if len(parts) == 2 and type(parts[1]) == str and type(parts[0].lower()) == str:
                    ipa_data[parts[0].lower()] = parts[1]
        return ipa_data
    def find_ipa(word, ipa_data):
        for entry in ipa_data:
            if entry[0] == word:
                return entry[1]
        return "IPA not found"
    de_filepath = 'data/monolingual_IPA_data/de.txt'
    nl_filepath = 'data/monolingual_IPA_data/nl.txt'
    en_filepath = 'data/monolingual_IPA_data/en_US.txt'
    sv_filepath = 'data/monolingual_IPA_data/sv.txt'
    de_ipa_data = load_ipa_data(de_filepath)
    nl_ipa_data = load_ipa_data(nl_filepath)
    en_ipa_data = load_ipa_data(en_filepath)
    sv_ipa_data = load_ipa_data(sv_filepath)
    data = {
        'de': de_ipa_data,
        'nl': nl_ipa_data,
        'en': en_ipa_data,
        'sv': sv_ipa_data
    }
    for name, value in data.items():
        print(f"Length of '{name}' is : {len(value)}")
    if cleanup:
        chars = set()
        for lang in data.values():
            for char in lang.values():
                chars.update(set(char))           
        chars = set(chars)
        pretty_chars = "pɡlɧɸɔɺθʁɪʈwɕʏɤɜrɑaɱɛɦæmãnɖœðʔʎsjfɬʉʙɭʕɨʊʀʒĭɹŋɯobəøɘʂäɥɲuạyɾɫtʤhdɚɵçkiɳzɒgʋɐxɝɣvõħʃeɓ"
        ugly_chars = ""
        for x in list(chars):
            if x not in pretty_chars:
                ugly_chars += x       
        for lang in data.values():
            for key in lang.keys():
                lang[key] = ''.join([char for char in lang[key] if char not in ugly_chars])
        print(f"Removed following chars: ({ugly_chars})") 
    return data


def convert_to_ipa(cognate_pairs: List[List[str]], ipa_lookup: Dict, lang_1: str, lang_2: str):
    converted = []
    for a, b in cognate_pairs:
        if a.lower() in ipa_lookup[lang_1].keys() and b.lower() in ipa_lookup[lang_2].keys():
            converted.append([ipa_lookup[lang_1][a.lower()], ipa_lookup[lang_2][b.lower()]])
        pass
    print(f"IPA Transliteration complete.\n\nRemaining cognates: {len(converted)}")
    return converted


