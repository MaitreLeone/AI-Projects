from datetime import datetime
from spacy.tokens import Doc as spacy_doc
import regex as re
import stanza
import copy
import trankit
import spacy
import pymorphy2
import string
import sys
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    MorphVocab,
    Doc,
)

# ---------------- SETTINGS SECTION ----------------

# set True if you need to use model for specific language
need_load_model_ru = False
need_load_model_en = False
need_load_model_es = False
need_load_model_fr = False
need_load_model_de = False
need_load_model_uk = False
need_load_model_tr = False
need_load_model_ar = False

# --------------- LOAD MODELS ----------------

nlp_morph = {}
nlp_token = {}

method = {
    'en': 'Stanza',
    'ru': 'Pymorphy',
    'es': 'Stanza',
    'fr': 'Stanza',
    'de': 'Stanza',
    'uk': 'Stanza',
    'tr': 'Stanza',
    'ar': 'Stanza',
}

if need_load_model_en:
    if method['en'] == 'Stanza':
        nlp_morph['en'] = stanza.Pipeline(lang="en", package='ewt', processors="tokenize,mwt,pos,lemma",
                                          tokenize_pretokenized=True)
        nlp_token['en'] = stanza.Pipeline(lang="en", package='ewt', processors="tokenize")
    elif method['en'] == 'Trankit':
        nlp_morph['en'] = trankit.Pipeline('english')
        nlp_token['en'] = trankit.Pipeline('english')
    elif method['en'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_morph['en'] = spacy.load("en_core_web_sm")
        nlp_token['en'] = spacy.load("en_core_web_sm")

if need_load_model_ru:
    if method['ru'] == 'Stanza':
        nlp_morph['ru'] = stanza.Pipeline(lang="ru", package='syntagrus', processors="tokenize,pos,lemma",
                                        tokenize_pretokenized=True)
        nlp_token['ru'] = stanza.Pipeline(lang="ru", package='gsd', processors="tokenize")
    elif method['ru'] == 'Trankit':
        nlp_morph['ru'] = trankit.Pipeline('russian')
        nlp_token['ru'] = trankit.Pipeline('russian')
    elif method['ru'] == 'Natasha':
        segmenter = Segmenter()
        morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb)
    elif method['ru'] == 'Pymorphy':
        '''
        'ADJF' (имя прил. полное) -> 'ADJ' (имя прил.)
        'ADJS' (имя прил. краткое) -> 'ADJ' (имя прил.)
        'COMP' (сравн. степень прил.) -> 'ADV' (наречие)
        'ADVB' (наречие) -> 'ADV' (наречие)
        'PREP' (предлог) -> 'ADP' (предлог)
        'GRND' (деепричастие) -> 'AUX' (вспомогательный глагол)
        'CONJ' (союз) -> 'CCONJ' (союз)
        'NUMR' (числительное) -> 'NUM' (числительное)
        'PRCL' (частица) -> 'PART' (частица)
        'NPRO' (местоимение-существительное) -> 'PRON' (местоимение)
        'INFN' (инфинитив) -> 'VERB' (глагол)
        'PRTF' (причастие полное) -> 'VERB' (глагол)
        'PRTS' (причастие краткое) -> 'VERB' (глагол)
        '''
        pos_list = ['ADJF', 'ADJS', 'COMP', 'ADVB', 'PREP', 'GRND', 'CONJ', 'NUMR', 'PRCL', 'NPRO', 'INFN', 'PRTF', 'PRTS']
        upos_list = ['ADJ', 'ADJ', 'ADV', 'ADV', 'ADP', 'AUX', 'CCONJ', 'NUM', 'PART', 'PRON', 'VERB', 'VERB', 'VERB']
        tenses_dict = {'past': 'past', 'pres': 'pres', 'futr': 'fut'}
        nlp_token['ru'] = stanza.Pipeline(lang="ru", package='gsd', processors="tokenize")
        nlp_morph_pymorphy = pymorphy2.MorphAnalyzer(lang='ru')

if need_load_model_es:
    if method['es'] == 'Stanza':
        nlp_morph['es'] = stanza.Pipeline(lang="es", package='gsd', processors="tokenize,mwt,pos,lemma",
                                          tokenize_pretokenized=True)
        nlp_token['es'] = stanza.Pipeline(lang="es", package='gsd', processors="tokenize")
    elif method['es'] == 'Trankit':
        nlp_morph['es'] = trankit.Pipeline('spanish')
        nlp_token['es'] = trankit.Pipeline('spanish')
    elif method['es'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_morph['es'] = spacy.load("es_core_news_sm")
        nlp_token['es'] = spacy.load("es_core_news_sm")

if need_load_model_fr:
    if method['fr'] == 'Stanza':
        nlp_morph['fr'] = stanza.Pipeline(lang="fr", package='gsd', processors="tokenize,mwt,pos,lemma",
                                          tokenize_pretokenized=True)
        nlp_token['fr'] = stanza.Pipeline(lang="fr", package='gsd', processors="tokenize")
    elif method['fr'] == 'Trankit':
        nlp_morph['fr'] = trankit.Pipeline('french')
        nlp_token['fr'] = trankit.Pipeline('french')
    elif method['fr'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_morph['fr'] = spacy.load("fr_core_news_sm")
        nlp_token['fr'] = spacy.load("fr_core_news_sm")

if need_load_model_de:
    if method['de'] == 'Stanza':
        nlp_morph['de'] = stanza.Pipeline(lang="de", package='gsd', processors="tokenize,mwt,pos,lemma",
                                          tokenize_pretokenized=True)
        nlp_token['de'] = stanza.Pipeline(lang="de", package='gsd', processors="tokenize")
    elif method['de'] == 'Trankit':
        nlp_morph['de'] = trankit.Pipeline('german')
        nlp_token['de'] = trankit.Pipeline('german')
    elif method['de'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_morph['de'] = spacy.load("de_core_news_sm")
        nlp_token['de'] = spacy.load("de_core_news_sm")

if need_load_model_uk:
    if method['uk'] == 'Stanza':
        nlp_morph['uk'] = stanza.Pipeline(lang="uk", package='iu', processors="tokenize,mwt,pos,lemma",
                                          tokenize_pretokenized=True)
        nlp_token['uk'] = stanza.Pipeline(lang="uk", package='iu', processors="tokenize")
    elif method['uk'] == 'Trankit':
        nlp_morph['uk'] = trankit.Pipeline('ukrainian')
        nlp_token['uk'] = trankit.Pipeline('ukrainian')
    elif method['uk'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_morph['uk'] = spacy.load("uk_core_news_sm")
        nlp_token['uk'] = spacy.load("uk_core_news_sm")

if need_load_model_tr:
    if method['tr'] == 'Stanza':
        nlp_morph['tr'] = stanza.Pipeline(lang="tr", package='imst', processors="tokenize,mwt,pos,lemma",
                                          tokenize_pretokenized=True)
        nlp_token['tr'] = stanza.Pipeline(lang="tr", package='imst', processors="tokenize")
    elif method['tr'] == 'Trankit':
        nlp_morph['tr'] = trankit.Pipeline('turkish')
        nlp_token['tr'] = trankit.Pipeline('turkish')

if need_load_model_ar:
    if method['ar'] == 'Stanza':
        nlp_morph['ar'] = stanza.Pipeline(lang="ar", package='padt', processors="tokenize,mwt,pos,lemma",
                                          tokenize_pretokenized=True)
        nlp_token['ar'] = stanza.Pipeline(lang="ar", package='padt', processors="tokenize")
    elif method['ar'] == 'Trankit':
        nlp_morph['ar'] = trankit.Pipeline('arabic')
        nlp_token['ar'] = trankit.Pipeline('arabic')

# --------------------------------------------------


def replace_symbol(token, symbol, tokens_list):
    token_splitted = re.split(re.escape(symbol), token)
    if len(token_splitted) > 1 and len(token) > 1:
        # split tokens, e.g. "Интерфакс\""
        token_splitted = token.replace(symbol, ' ' + symbol + ' ').strip().split()
        for item in token_splitted:
            if len(item) > 0:
                tokens_list.append([item, '', '', '', -1, -1])
    else:
        tokens_list.append([token, '', '', '', -1, -1])
    return tokens_list


def append_token(token, tokens_list):
    if '\"' in token:
        tokens_list = replace_symbol(token, '\"', tokens_list)
    # chr(0x2D) = '-'
    elif token.startswith(chr(0x2D)) or token.endswith(chr(0x2D)):
        tokens_list = replace_symbol(token, chr(0x2D), tokens_list)
    # chr(0x2013) = '–'
    elif token.startswith(chr(0x2013)) or token.endswith(chr(0x2013)):
        tokens_list = replace_symbol(token, chr(0x2013), tokens_list)
    # chr(0x2014) = '—'
    elif token.startswith(chr(0x2014)) or token.endswith(chr(0x2014)):
        tokens_list = replace_symbol(token, chr(0x2014), tokens_list)
    elif token.endswith('.'):
        tokens_list = replace_symbol(token, '.', tokens_list)
    elif token.endswith(':'):
        tokens_list = replace_symbol(token, ':', tokens_list)
    elif token.endswith('…'):
        tokens_list = replace_symbol(token, '…', tokens_list)
    elif token.endswith('?'):
        tokens_list = replace_symbol(token, '?', tokens_list)
    elif token.endswith('!'):
        tokens_list = replace_symbol(token, '!', tokens_list)
    elif token.startswith('('):
        tokens_list = replace_symbol(token, '(', tokens_list)
        if token.endswith(')'):
            tokens_list = replace_symbol(tokens_list[-1][0], ')', tokens_list[:-1])
    elif token.endswith(')'):
        tokens_list = replace_symbol(token, ')', tokens_list)
    elif token.startswith('['):
        tokens_list = replace_symbol(token, '[', tokens_list)
        if token.endswith(']'):
            tokens_list = replace_symbol(tokens_list[-1][0], ']', tokens_list[:-1])
    elif token.endswith(']'):
        tokens_list = replace_symbol(token, ']', tokens_list)
    elif token.startswith('{'):
        tokens_list = replace_symbol(token, '{', tokens_list)
        if token.endswith('}'):
            tokens_list = replace_symbol(tokens_list[-1][0], '}', tokens_list[:-1])
    elif token.endswith('}'):
        tokens_list = replace_symbol(token, '}', tokens_list)
    else:
        tokens_list.append([token, '', '', '', -1, -1])
    return tokens_list


def get_tokens_stanza(doc):
    tokens_list = []
    for sentence in doc.sentences:
        for token in sentence.tokens:
            tokens_list = append_token(token.text, tokens_list)
    return tokens_list


def get_tokens_trankit(tokens):
    tokens_list = []
    for token in tokens['tokens']:
        tokens_list = append_token(token['text'], tokens_list)
    return tokens_list


def get_tokens_spacy(doc):
    tokens_list = []
    for token in doc:
        tokens_list = append_token(token.text, tokens_list)
    return tokens_list

def get_tokens_natasha(doc):
    tokens_list = []
    doc.segment(segmenter)
    for token in doc.tokens:
        tokens_list = append_token(token.text, tokens_list)
    return tokens_list

def get_tokens_from_grammar(message):
    """
    Takes one document from text corpus
    and extract tokens from field 'grammar'

    Returns list sentences, every sentence
    contains tokens from field 'grammar'
    """

    tokens_list = []
    for sentence in message['sentences']:
        temp_list = []
        for token in sentence['grammar']:
            temp_list.append(token[0])  # token is list looking like [wordtoken, ...]
        tokens_list.append(temp_list)

    return tokens_list


def get_morph_stanza(sentence, grammar, message_id):
    morph_list = []
    for i in range(len(sentence.words)):
        word = sentence.words[i]
        word_feats = ''
        if word.upos == 'VERB' and word.feats is not None:
            j = 0
            feats = word.feats.lower().split('|')
            while j < len(feats) and not feats[j].startswith('tense'):
                j += 1
            if j < len(feats):
                word_feats = feats[j]
        elif word.upos == 'NOUN' and word.feats is not None:
            feats = word.feats.lower().split('|')
            j = 0
            while j < len(feats) and not feats[j].startswith('number'):
                j += 1
            if j < len(feats):
                word_feats = feats[j]
            j = 0
            while j < len(feats) and not feats[j].startswith('animacy'):
                j += 1
            if j < len(feats):
                if len(word_feats):
                    word_feats += '|'
                word_feats += feats[j]
        if word.lemma is not None:
            morph_list.append([word.text, word.lemma.lower(), word.upos, word_feats, grammar[i][4], grammar[i][5]])
        else:
            morph_list.append([word.text, word.text.lower(), word.upos, word_feats, grammar[i][4], grammar[i][5]])
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print('%s Text ID: %s, Warning: lemma of %s is undefined' % (t, message_id, word.text))
    return morph_list


def get_morph_trankit(tagged_sentence, lemmatized_sentence, grammar, message_id):
    morph_list = []
    token_text_list = []
    token_upos_list = []
    token_lemma_list = []
    token_feats_list = []
    for token in tagged_sentence['tokens']:    
        word_feats = ''
        if token['upos'] == 'VERB' and token['feats'] is not None:
            j = 0
            feats = token['feats'].lower().split('|')
            while j < len(feats) and not feats[j].startswith('tense'):
                j += 1
            if j < len(feats):
                word_feats = feats[j]
        elif token['upos'] == 'NOUN' and token['feats'] is not None:
            feats = token['feats'].lower().split('|')
            j = 0
            while j < len(feats) and not feats[j].startswith('number'):
                j += 1
            if j < len(feats):
                word_feats = feats[j]
            j = 0
            while j < len(feats) and not feats[j].startswith('animacy'):
                j += 1
            if j < len(feats):
                if len(word_feats):
                    word_feats += '|'
                word_feats += feats[j]
        token_text_list.append(token['text'])
        token_upos_list.append(token['upos'])
        token_feats_list.append(word_feats)
    for token in lemmatized_sentence['tokens']:
        if token['lemma'] is not None:
            token_lemma_list.append(token['lemma'].lower())
        else:
            token_lemma_list.append(token['text'].lower())
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print('%s Text ID: %s, Warning: lemma of %s is undefined' % (t, message_id, token['text']))
    for i in range(len(grammar)):
        morph_list.append([token_text_list[i], token_lemma_list[i], token_upos_list[i], token_feats_list[i],
                           grammar[i][4], grammar[i][5]])
    return morph_list


def get_morph_spacy(doc, grammar, message_id):
    morph_list = []
    for i, token in enumerate(doc):
        word_feats = ''
        if token.pos_ == 'VERB' and token.morph is not None:
            j = 0
            feats = str(token.morph).lower().split('|')
            while j < len(feats) and not feats[j].startswith('tense'):
                j += 1
            if j < len(feats):
                word_feats = feats[j]
        elif token.pos_ == 'NOUN' and token.morph is not None:
            feats = str(token.morph).lower().split('|')
            j = 0
            while j < len(feats) and not feats[j].startswith('number'):
                j += 1
            if j < len(feats):
                word_feats = feats[j]
            j = 0
            while j < len(feats) and not feats[j].startswith('animacy'):
                j += 1
            if j < len(feats):
                if len(word_feats):
                    word_feats += '|'
                word_feats += feats[j]
        if token.lemma_ is not None:
            morph_list.append([token.text, token.lemma_.lower(), token.pos_, word_feats, grammar[i][4], grammar[i][5]])
        else:
            morph_list.append([token.text, token.text.lower(), token.pos_, word_feats, grammar[i][4], grammar[i][5]])
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print('%s Text ID: %s, Warning: lemma of %s is undefined' % (t, message_id, token.text))
    return morph_list


def get_morph_natasha(doc, grammar, message_id):
    morph_list = []
    for i in range(len(doc.tokens)):
        token = doc.tokens[i]
        token.lemmatize(morph_vocab)
        word_feats = ''
        if token.pos == 'VERB' and token.feats is not None:
            feats = ''
            for k,v in token.feats.items():
                if k == 'Tense':
                    feats += str(k).lower() + '=' + str(v).lower()
            word_feats = feats
        elif token.pos == 'NOUN' and token.feats is not None:
            feats = ''
            for k,v in token.feats.items():
                if k == 'Number' or k == 'Animacy':
                    if len(feats) > 0:
                        feats += '|'
                    feats += str(k).lower() + '=' + str(v).lower()
            word_feats += feats
        if token.lemma is not None:
            morph_list.append([token.text, token.lemma.lower(), token.pos, word_feats, grammar[i][4], grammar[i][5]])
        else:
            morph_list.append([token.text, token.text.lower(), token.pos, word_feats, grammar[i][4], grammar[i][5]])
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print('%s Text ID: %s, Warning: lemma of %s is undefined' % (t, message_id, token.text))
    return morph_list


def get_morph_pymorphy(grammar, message_id):
    morph_list = []
    for word in grammar:
        morph = nlp_morph_pymorphy.parse(word[0])[0]
        upos = morph.tag.POS
        if morph.normal_form is not None:
            # print('morph tag:', morph.tag)
            lemma = morph.normal_form.lower().replace('ё', 'е')
            if word[0] in string.punctuation + '–—':
                morph_list.append([word[0], lemma, 'PUNCT', word[3], word[4], word[5]])
            elif upos is not None:
                if upos in pos_list:
                    upos = upos_list[pos_list.index(upos)]
                
                if upos == 'NOUN':
                    for tag in ['Surn', 'Patr', 'Name', 'Geox']:
                        if tag in morph.tag:
                            upos = 'PROPN'
                            break
                
                feats = ''
                if upos == 'VERB' and morph.tag.tense is not None:
                    feats = 'tense='
                    if morph.tag.tense == 'futr':
                        feats += 'fut'
                    else:
                        feats += morph.tag.tense
                elif upos == 'NOUN' and morph.tag.number is not None and morph.tag.animacy is not None:
                    feats = 'number=' + morph.tag.number + '|animacy=' + morph.tag.animacy
                morph_list.append([word[0], lemma, upos, feats, word[4], word[5]])
            elif sum([1 if ch.isalpha() else 0 for ch in word[0]]) == 0 and \
                    sum([1 if ch.isdigit() else 0 for ch in word[0]]) > 0:
                morph_list.append([word[0], lemma, 'NUM', word[3], word[4], word[5]])
            else:
                morph_list.append([word[0], lemma, 'X', word[3], word[4], word[5]])
        else:
            morph_list.append([word[0], word[0].lower(), upos, word[3], word[4], word[5]])
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print('%s Text ID: %s, Warning: lemma of %s is undefined' % (t, message_id, word[0]))
    return morph_list


def correct_morph(grammar, text, method_name, nlp_morph):
    '''
    Get the lemma of the first word of the letter digit complexes

    Returns grammar with corrected lemma of the letter digit complexes
    '''
    # Regular expressions for экс-премьер-министр, формула-1
    letter_digit_complex_patterns = [r'\pL+-\pL*[0-9]*\pL*-\pL*[0-9]*\pL*', r'\pL+-\pL*[0-9]*\pL*']

    finds = []
    for pattern in letter_digit_complex_patterns:
        finds.extend(re.findall(pattern, text))
    if len(finds) == 0:
        return grammar

    tokens = [item[0] for item in grammar]
    is_digit = False
    for complex in finds:
        if complex in tokens:
            if complex.split('-')[1].isdigit():
                is_digit = True
                word = complex.split('-')[0]
            else:
                word = complex.split('-')[1]
            if word.isupper():
                continue
            lemma = ''
            if method_name == 'Stanza':
                doc = nlp_morph(word)
                lemma = doc.sentences[0].words[0].lemma
            elif method_name == 'Trankit':
                lemma = nlp_morph.lemmatize(word, is_sent=True)['tokens'][0]['lemma']
            elif method_name == 'Spacy':
                doc = nlp_morph(word)
                lemma = doc[0].lemma_
            elif method_name == 'Natasha':
                doc_complex = Doc(word)
                doc_complex.segment(segmenter)
                doc_complex.tag_morph(morph_tagger)
                doc_complex.tokens[0].lemmatize(morph_vocab)
                lemma = doc_complex.tokens[0].lemma
            elif method_name == 'Pymorphy':
                morph = pymorphy2.MorphAnalyzer()
                lemma = morph.parse(word)[0].normal_form
            if is_digit:
                complex_new = lemma + '-' + '-'.join(complex.split('-')[1:])
            else:
                complex_new = complex.split('-')[0] + '-' + lemma + '-'.join(complex.split('-')[2:])
            complex_new = complex_new.lower()
            if tokens.count(complex) == 1:
                complex_index = tokens.index(complex)
                grammar[complex_index][1] = complex_new
            elif tokens.count(complex) > 1:
                for i_complex in range(len(tokens)):
                    if tokens[i_complex] == complex:
                        grammar[i_complex][1] = complex_new

    return grammar


def join_tokens(text, grammar):
    # Regular expressions for
    # Ту-204-300, Миг-29,
    # экс-премьер-министр, премьер-министр, 2019-nCoV, 2019-2020,
    # 5,8%, $20, О'Нил,
    # Газета.ру, www.twitter.com,
    # Е.Н.О.Т., д.т.н., A.B., А.
    letter_digit_complex_patterns = [r'\pL+-\pL*[0-9]+\pL*-\pL*[0-9]+\pL*', r'\pL+-\pL*[0-9]+\pL*',
                                     r'\pL+-\pL+-\pL+', r'\pL+-\pL+', '[0-9]+-\pL+', r'[0-9]+-[0-9]+',
                                     r'[0-9]+,?[0-9]*%', r'\$[0-9]+,?[0-9]*', r'\p{Lu}\'\pL+',
                                     r'\pL+\.\pL+', r'\pL+\.\pL+\.\pL+',
                                     r'\pL\.\pL\.\pL\.\pL\.', r'\pL\.\pL\.\pL\.', r'\pL\.\pL\.', r'\p{Lu}\.']

    finds = []
    for pattern in letter_digit_complex_patterns:
        finds.extend(re.findall(pattern, text))
    finds = sorted(list(set(finds)), key=len, reverse=True)

    tokens = [item[0] for item in grammar]
    for elem in finds:
        if elem not in tokens:
            if '-' in elem:
                complex_len = elem.count('-') * 2 + 1
                for i in range(len(tokens) - complex_len + 1):
                    tokens_joined = ''.join(tokens[i:i + complex_len])
                    if tokens_joined == elem:
                        tokens[i] = tokens_joined
                        grammar[i][0] = tokens_joined
                        del grammar[i + 1: i + complex_len]
                        del tokens[i + 1: i + complex_len]
            if '%' in elem or '$' in elem:
                complex_len = 2
                for i in range(len(tokens) - complex_len + 1):
                    tokens_joined = ''.join(tokens[i:i + complex_len])
                    if tokens_joined == elem:
                        tokens[i] = tokens_joined
                        grammar[i][0] = tokens_joined
                        del grammar[i + 1: i + complex_len]
                        del tokens[i + 1: i + complex_len]
            elif '.' in elem:
                complex_len = elem.count('.') * 2
                stop = False
                for complex_len_temp in range(complex_len, 1, -1):
                    for i in range(len(tokens) - complex_len_temp + 1):
                        tokens_joined = ''.join(tokens[i:i + complex_len_temp])
                        if tokens_joined == elem:
                            stop = True
                            tokens[i] = tokens_joined
                            grammar[i][0] = tokens_joined
                            del grammar[i + 1: i + complex_len_temp]
                            del tokens[i + 1: i + complex_len_temp]
                    if stop:
                        break

    return grammar


def remove_symbols_ent(lemmas_ent):
    # Remove symbols \"[]() from lemma of the entity
    for symbol in '\"[]()':
        if symbol in lemmas_ent:
            num_symbols = lemmas_ent.count(symbol)
            for i in range(num_symbols):
                lemmas_ent.remove(symbol)
    return lemmas_ent


def remove_symbols(entity, lemmas):
    # Remove symbols \"[]() from lemma of the entity
    for symbol in '\"[]()':
        start_grammar = entity['entries'][0]['start_grammar']
        end_grammar = entity['entries'][0]['end_grammar']
        if lemmas[start_grammar] == symbol:
            for i in range(len(entity['entries'])):
                entity['entries'][i]['start_grammar'] += 1
        if lemmas[end_grammar] == symbol:
            for i in range(len(entity['entries'])):
                entity['entries'][i]['end_grammar'] -= 1
        start_grammar = entity['entries'][0]['start_grammar']
        end_grammar = entity['entries'][0]['end_grammar']
        lemmas_ent = lemmas[start_grammar: end_grammar + 1]
        if symbol in lemmas_ent:
            num_symbols = lemmas_ent.count(symbol)
            for i in range(num_symbols):
                lemmas_ent.remove(symbol)
    return lemmas_ent


def calc_ner_weight(message):
    sentences = message['sentences']

    ner_freq = {}
    for i in range(len(sentences)):
        if 'entities' in sentences[i]:
            for ent_lemma in sentences[i]['entities']:
                if ent_lemma not in ner_freq:
                    ner_freq[ent_lemma] = len(sentences[i]['entities'][ent_lemma]['entries'])
                else:
                    ner_freq[ent_lemma] += len(sentences[i]['entities'][ent_lemma]['entries'])

    if 'title_analysis' in message:
        if 'entities' in message['title_analysis']:
            for ent_lemma in message['title_analysis']['entities']:
                if ent_lemma not in ner_freq:
                    ner_freq[ent_lemma] = len(message['title_analysis']['entities'][ent_lemma]['entries'])
                else:
                    ner_freq[ent_lemma] += len(message['title_analysis']['entities'][ent_lemma]['entries'])
    
    for i in range(len(sentences)):
        if 'entities' in sentences[i]:
            for ent_lemma in sentences[i]['entities']:
                sentences[i]['entities'][ent_lemma]['weight'] = ner_freq[ent_lemma]

    if 'title_analysis' in message:
        if 'entities' in message['title_analysis']:
            for ent_lemma in message['title_analysis']['entities']:
                message['title_analysis']['entities'][ent_lemma]['weight'] = ner_freq[ent_lemma]


def change_ner_format(ner):
    dates = {}
    entities = {}
    for ner_type in ner.keys():
        new_ner = {}
        for value in list(ner[ner_type].values()):
            if value['lemma'] not in new_ner.keys():
                new_ner[value['lemma']] = {'entries': value['entries']}
            else:
                for elem in value['entries']:
                    new_ner[value['lemma']]['entries'].append(elem)
        for entity_lemma in new_ner:
            entries = new_ner[entity_lemma]['entries']
            new_ner[entity_lemma]['entries'] = sorted(entries, key=lambda x: x['start'])
            if ner_type == 'DATE':
                dates[entity_lemma] = new_ner[entity_lemma]
            else:
                entities[entity_lemma] = new_ner[entity_lemma]
                entities[entity_lemma]['type'] = 'ner'
                entities[entity_lemma]['class'] = ner_type
    return entities, dates


def morph_analysis(messages):
    """ Функция морфологического анализа входных сообщений.
    
    Морфологический анализ осуществляется для отдельных предложений (поле `sentences`) и для заголовка (поля `title` и `title_analysis`).
    
    На вход функции предложения и заголовок приходят токенизированными (есть поля `grammar` с выделенными словоформами).
    
    :param messages: список входных сообщений в формате JSON (dict). Каждое сообщение содержит поля:
            - `lang` - язык;
            - `sentences` - список предложений. Каждое предложение представляет собой словарь, содержащий поля^:
                - `text` - исходный текст предложения;
                - `grammar` - список слов (списков), для каждого слова имеется выделенная (токенизированная) словоформа;
                - `ner` - перечень именованных сущностей;
            - `title` - заголовок сообщения (опционально).
            - `title_analysis` - результаты анализа заголовка (опционально) - словарь, в котором имеются поля:
                - `grammar` - аналогичное по смыслу полю `grammar` для предложений;
                - `ner` - аналогичное по смыслу полю `ner` для предложений.
    
    :return:
        список входных сообщений, для каждого из которых:
            - для каждого предложения (в поле `sentences`) в поле `grammar` добавлены результаты морфологического анализа. Это список списков; каждый вложенный список включает 6 элементов:
                - исходная словоформа (регистрозависимая) - не меняется функцией;
                - нормальная форма слова, приведенная к нижнему регистру - заполняется функцией;
                - часть речи - заполняется функцией;
                - время глагола (в случае отсутствия - пустая строка) - не меняется функцией;
                - ID данного слова в предложении, начиная с единицы - не меняется функцией;
                - ID слова, с которым связано данное слово - не меняется функцией;
            - для каждого предложения (в поле `sentences`) - для каждой именованной сущности (поле `ner`) заполняется элемент списка, соответствующий нормальной форме именованной сущности;
            - в результаты анализа заголовка (поле `title_analysis`) в поле `grammar` добавлены результаты морфологического анализа - аналогичные по смыслу полю `grammar` для предложений;
            - в результаты анализа заголовка (поле `title_analysis`) - для каждой именованной сущности (поле `ner`) заполняется элемент списка, соответствующий нормальной форме именованной сущности.
    
    """

    messages_new = copy.deepcopy(messages)

    for message in messages_new:
        message_id = ''
        if 'id' in message.keys():
            message_id = message['id']
        if 'lang' not in message or 'sentences' not in message or len(message['sentences']) == 0:
            break

        sentences = message['sentences']
        try:
            tokens = get_tokens_from_grammar(message)
            if method[message['lang']] == 'Stanza':
                doc = nlp_morph[message['lang']](tokens) # let pretokeinzed text to model
                for i in range(len(sentences)):
                    grammar = get_morph_stanza(doc.sentences[i], sentences[i]['grammar'], message_id)
                    sentences[i]['grammar'] = correct_morph(grammar, sentences[i]['text'],
                                                            method[message['lang']], nlp_morph[message['lang']])
            elif method[message['lang']] == 'Trankit':
                for i in range(len(sentences)):
                    tagged_sent = nlp_morph[message['lang']].posdep(tokens[i], is_sent=True)
                    lemmatized_sent = nlp_morph[message['lang']].lemmatize(tokens[i], is_sent=True)
                    grammar = get_morph_trankit(tagged_sent, lemmatized_sent, sentences[i]['grammar'], message_id)
                    sentences[i]['grammar'] = correct_morph(grammar, sentences[i]['text'],
                                                            method[message['lang']], nlp_morph[message['lang']])
            elif method[message['lang']] == 'Spacy':
                docs = [spacy_doc(nlp_morph[message['lang']].vocab, tokens[i]) for i in range(len(tokens))]
                docs = nlp_morph[message['lang']].pipe(docs)
                for i, doc in enumerate(docs):
                    grammar = get_morph_spacy(doc, sentences[i]['grammar'], message_id)
                    sentences[i]['grammar'] = correct_morph(grammar, sentences[i]['text'],
                                                            method[message['lang']], nlp_morph[message['lang']])
            elif method[message['lang']] == 'Natasha' and message['lang'] == 'ru':
                for i in range(len(sentences)):
                    doc = Doc(sentences[i]['text'])
                    doc.segment(segmenter)
                    doc.tag_morph(morph_tagger)
                    grammar = get_morph_natasha(doc, sentences[i]['grammar'], message_id)
                    sentences[i]['grammar'] = correct_morph(grammar, sentences[i]['text'],
                                                            method[message['lang']], '')
            elif method[message['lang']] == 'Pymorphy' and message['lang'] == 'ru':
                for i in range(len(sentences)):
                    grammar = get_morph_pymorphy(sentences[i]['grammar'], message_id)
                    sentences[i]['grammar'] = correct_morph(grammar, sentences[i]['text'],
                                                            method[message['lang']], '')
            for i in range(len(sentences)):
                if 'ner' in sentences[i]:
                    for ner_type in sentences[i]['ner'].keys():
                        for ent in sentences[i]['ner'][ner_type].keys():
                            lemmas = [item[1] for item in sentences[i]['grammar']]

                            start_grammar = -1
                            end_grammar = -1
                            if len(sentences[i]['ner'][ner_type][ent]['entries']) > 0:
                                start_grammar = sentences[i]['ner'][ner_type][ent]['entries'][0]['start_grammar']
                                end_grammar = sentences[i]['ner'][ner_type][ent]['entries'][0]['end_grammar']
                            if start_grammar != -1 and end_grammar != -1:
                                lemmas_ner = remove_symbols(sentences[i]['ner'][ner_type][ent], lemmas)
                                sentences[i]['ner'][ner_type][ent]['lemma'] = ' '.join(lemmas_ner)
                            else:
                                if method[message['lang']] == 'Stanza':
                                    grammar_ner = get_tokens_stanza(nlp_token[message['lang']](ent))
                                    grammar_ner = join_tokens(sentences[i]['text'], grammar_ner)
                                    tokens_ner = [item[0] for item in grammar_ner]
                                    doc_ner = nlp_morph[message['lang']]([tokens_ner])
                                    lemmas_ner = [item[1] for item in get_morph_stanza(doc_ner.sentences[0], grammar_ner, message_id)]
                                    sentences[i]['ner'][ner_type][ent]['lemma'] = ' '.join(remove_symbols_ent(lemmas_ner))
                                elif method[message['lang']] == 'Trankit':
                                    grammar_ner = get_tokens_trankit(nlp_token[message['lang']](ent))
                                    grammar_ner = join_tokens(sentences[i]['text'], grammar_ner)
                                    tokens_ner = [item[0] for item in grammar_ner]
                                    doc_ner = nlp_morph[message['lang']]([tokens_ner])
                                    lemmas_ner = [item[1] for item in get_morph_trankit(doc_ner['tokens'], grammar_ner, message_id)]
                                    sentences[i]['ner'][ner_type][ent]['lemma'] = ' '.join(remove_symbols_ent(lemmas_ner))
                                elif method[message['lang']] == 'Spacy':
                                    grammar_ner = get_tokens_spacy(nlp_token[message['lang']](ent))
                                    grammar_ner = join_tokens(sentences[i]['text'], grammar_ner)
                                    tokens_ner = [item[0] for item in grammar_ner]
                                    doc_ner = spacy_doc(nlp_morph[message['lang']].vocab, tokens_ner)
                                    doc_ner = nlp_morph[message['lang']](doc_ner)
                                    lemmas_ner = [item[1] for item in get_morph_spacy(doc_ner, grammar_ner, message_id)]
                                    sentences[i]['ner'][ner_type][ent]['lemma'] = ' '.join(remove_symbols_ent(lemmas_ner))
                                elif method[message['lang']] == 'Natasha' and message['lang'] == 'ru':
                                    doc_ner = Doc(ent)
                                    doc_ner.segment(segmenter)
                                    doc_ner.tag_morph(morph_tagger)
                                    grammar_ner = [[token.text, '', '', '', -1, -1] for token in doc_ner.tokens]
                                    grammar_ner = join_tokens(sentences[i]['text'], grammar_ner)
                                    lemmas_ner = [item[1] for item in get_morph_natasha(doc_ner, grammar_ner, message_id)]
                                    sentences[i]['ner'][ner_type][ent]['lemma'] = ' '.join(remove_symbols_ent(lemmas_ner))
                                elif method[message['lang']] == 'Pymorphy' and message['lang'] == 'ru':
                                    grammar_ner = get_tokens_stanza(nlp_token[message['lang']](ent))
                                    grammar_ner = join_tokens(sentences[i]['text'], grammar_ner)
                                    lemmas_ner = [item[1] for item in get_morph_pymorphy(grammar_ner, message_id)]
                                    sentences[i]['ner'][ner_type][ent]['lemma'] = ' '.join(remove_symbols_ent(lemmas_ner))

                    entities, dates = change_ner_format(sentences[i]['ner'])
                    if len(entities.keys()) > 0:
                        sentences[i]['entities'] = entities
                    if len(dates.keys()) > 0:
                        sentences[i]['dates'] = dates
                    del sentences[i]['ner']
        except:
            e = sys.exc_info()[1]
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print("%s Module: %s, function: %s, text ID: %s, field: %s, error: %s" %
                  (t, 'morphology', 'morph_analysis', message_id, 'sentences', e))

        if 'title_analysis' in message:
            try:
                title_analysis = message['title_analysis']
                list_sentences = [{'text': message['title'], 'grammar': title_analysis['grammar']}]
                custom_message = {'sentences': list_sentences}
                tokens = get_tokens_from_grammar(custom_message)
                if method[message['lang']] == 'Stanza':
                    doc = nlp_morph[message['lang']](tokens)  # let pretokeinzed text to model
                    grammar = get_morph_stanza(doc.sentences[0], title_analysis['grammar'], message_id)
                    title_analysis['grammar'] = correct_morph(grammar, message['title'],
                                                              method[message['lang']], nlp_morph[message['lang']])
                elif method[message['lang']] == 'Trankit':
                    tagged_sent = nlp_morph[message['lang']].posdep(tokens[0], is_sent=True)
                    lemmatized_sent = nlp_morph[message['lang']].lemmatize(tokens[0], is_sent=True)
                    grammar = get_morph_trankit(tagged_sent, lemmatized_sent, title_analysis['grammar'], message_id)
                    title_analysis['grammar'] = correct_morph(grammar, message['title'],
                                                              method[message['lang']], nlp_morph[message['lang']])
                elif method[message['lang']] == 'Spacy':
                    doc = spacy_doc(nlp_morph[message['lang']].vocab, tokens[0])
                    doc = nlp_morph[message['lang']](doc)
                    grammar = get_morph_spacy(doc, title_analysis['grammar'], message_id)
                    title_analysis['grammar'] = correct_morph(grammar, message['title'],
                                                              method[message['lang']], nlp_morph[message['lang']])
                elif method[message['lang']] == 'Natasha' and message['lang'] == 'ru':
                    doc = Doc(message['title'])
                    doc.segment(segmenter)
                    doc.tag_morph(morph_tagger)
                    grammar = get_morph_natasha(doc, title_analysis['grammar'], message_id)
                    title_analysis['grammar'] = correct_morph(grammar, message['title'],
                                                              method[message['lang']], '')
                elif method[message['lang']] == 'Pymorphy' and message['lang'] == 'ru':
                    grammar = get_morph_pymorphy(title_analysis['grammar'], message_id)
                    title_analysis['grammar'] = correct_morph(grammar, message['title'],
                                                              method[message['lang']], '')

                if 'ner' in title_analysis and title_analysis['ner'] is not None:
                    for ner_type in title_analysis['ner'].keys():
                        for ent in title_analysis['ner'][ner_type].keys():
                            lemmas = [item[1] for item in title_analysis['grammar']]

                            start_grammar = -1
                            end_grammar = -1
                            if len(title_analysis['ner'][ner_type][ent]['entries']) > 0:
                                start_grammar = title_analysis['ner'][ner_type][ent]['entries'][0]['start_grammar']
                                end_grammar = title_analysis['ner'][ner_type][ent]['entries'][0]['end_grammar']
                            if start_grammar != -1 and end_grammar != -1:
                                lemmas_ner = remove_symbols(title_analysis['ner'][ner_type][ent], lemmas)
                                title_analysis['ner'][ner_type][ent]['lemma'] = ' '.join(lemmas_ner)
                            else:
                                if method[message['lang']] == 'Stanza':
                                    grammar_ner = get_tokens_stanza(nlp_token[message['lang']](ent))
                                    grammar_ner = join_tokens(message['title'], grammar_ner)
                                    tokens_ner = [item[0] for item in grammar_ner]
                                    doc_ner = nlp_morph[message['lang']]([tokens_ner])
                                    lemmas_ner = [item[1] for item in get_morph_stanza(doc_ner.sentences[0], grammar_ner, message_id)]
                                    title_analysis['ner'][ner_type][ent]['lemma'] = ' '.join(remove_symbols_ent(lemmas_ner))
                                elif method[message['lang']] == 'Trankit':
                                    grammar_ner = get_tokens_trankit(nlp_token[message['lang']](ent))
                                    grammar_ner = join_tokens(message['title'], grammar_ner)
                                    tokens_ner = [item[0] for item in grammar_ner]
                                    doc_ner = nlp_morph[message['lang']]([tokens_ner])
                                    lemmas_ner = [item[1] for item in get_morph_trankit(doc_ner['tokens'], grammar_ner, message_id)]
                                    title_analysis['ner'][ner_type][ent]['lemma'] = ' '.join(remove_symbols_ent(lemmas_ner))
                                elif method[message['lang']] == 'Spacy':
                                    grammar_ner = get_tokens_spacy(nlp_token[message['lang']](ent))
                                    grammar_ner = join_tokens(message['title'], grammar_ner)
                                    tokens_ner = [item[0] for item in grammar_ner]
                                    doc_ner = spacy_doc(nlp_morph[message['lang']].vocab, tokens_ner)
                                    doc_ner = nlp_morph[message['lang']](doc_ner)
                                    lemmas_ner = [item[1] for item in get_morph_spacy(doc_ner['tokens'], grammar_ner, message_id)]
                                    title_analysis['ner'][ner_type][ent]['lemma'] = ' '.join(remove_symbols_ent(lemmas_ner))
                                elif method[message['lang']] == 'Natasha' and message['lang'] == 'ru':
                                    doc_ner = Doc(ent)
                                    doc_ner.segment(segmenter)
                                    doc_ner.tag_morph(morph_tagger)
                                    grammar_ner = [[token.text, '', '', '', -1, -1] for token in doc_ner.tokens]
                                    grammar_ner = join_tokens(message['title'], grammar_ner)
                                    lemmas_ner = [item[1] for item in get_morph_natasha(doc_ner, grammar_ner, message_id)]
                                    title_analysis['ner'][ner_type][ent]['lemma'] = ' '.join(remove_symbols_ent(lemmas_ner))
                                elif method[message['lang']] == 'Pymorphy' and message['lang'] == 'ru':
                                    grammar_ner = get_tokens_stanza(nlp_token[message['lang']](ent))
                                    grammar_ner = join_tokens(message['title'], grammar_ner)
                                    lemmas_ner = [item[1] for item in get_morph_pymorphy(grammar_ner, message_id)]
                                    title_analysis['ner'][ner_type][ent]['lemma'] = ' '.join(remove_symbols_ent(lemmas_ner))

                    entities, dates = change_ner_format(title_analysis['ner'])
                    if len(entities.keys()) > 0:
                        title_analysis['entities'] = entities
                    if len(dates.keys()) > 0:
                        title_analysis['dates'] = dates
                    del title_analysis['ner']
            except:
                e = sys.exc_info()[1]
                t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print("%s Module: %s, function: %s, text ID: %s, field: %s, error: %s" %
                      (t, 'morphology', 'morph_analysis', message_id, 'title_analysis', e))

        try:
            calc_ner_weight(message)
        except:
            e = sys.exc_info()[1]
            t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print("%s Module: %s, function: %s, text ID: %s, error: %s" %
                  (t, 'morphology', 'calc_ner_weight', message_id, e))

    return messages_new
