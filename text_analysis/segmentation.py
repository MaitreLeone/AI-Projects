from nltk import regexp_tokenize, sent_tokenize
from datetime import datetime
import stanza
import string
import copy
import trankit
import spacy
import sys

from natasha import (
    Segmenter,
    Doc
)


# ---------------- SETTINGS SECTION ----------------

# set True if you need to use model for specific language
need_load_model_ru = True
need_load_model_en = False
need_load_model_es = False
need_load_model_fr = False
need_load_model_de = False
need_load_model_uk = False
need_load_model_tr = False
need_load_model_ar = False

# --------------- LOAD MODELS ----------------

nlp_token = {}

method = {
    'en': 'Stanza',
    'ru': 'Stanza',
    'es': 'Stanza',
    'fr': 'Stanza',
    'de': 'Stanza',
    'uk': 'Stanza',
    'tr': 'Stanza',
    'ar': 'Stanza'
}

if need_load_model_en:
    if method['en'] == 'Stanza':
        nlp_token['en'] = stanza.Pipeline(lang="en", package='ewt', processors='tokenize')
    elif method['en'] == 'Trankit':
        nlp_token['en'] = trankit.Pipeline(lang='english', embedding='xlm-roberta-large', gpu=True)
    elif method['en'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['en'] = spacy.load("en_core_web_sm")

if need_load_model_ru:
    if method['ru'] == 'Stanza':
        nlp_token['ru'] = stanza.Pipeline(lang="ru", package='gsd', processors='tokenize')
    elif method['ru'] == 'Trankit':
        nlp_token['ru'] = trankit.Pipeline(lang='russian', embedding='xlm-roberta-large', gpu=True)
    elif method['ru'] == 'Natasha':
        segmenter = Segmenter()

if need_load_model_es:
    if method['es'] == 'Stanza':
        nlp_token['es'] = stanza.Pipeline(lang="es", package='gsd', processors='tokenize')
    elif method['es'] == 'Trankit':
        nlp_token['es'] = trankit.Pipeline(lang='spanish', embedding='xlm-roberta-large', gpu=True)
    elif method['es'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['es'] = spacy.load("es_core_news_sm")

if need_load_model_fr:
    if method['fr'] == 'Stanza':
        nlp_token['fr'] = stanza.Pipeline(lang="fr", package='gsd', processors='tokenize')
    elif method['fr'] == 'Trankit':
        nlp_token['fr'] = trankit.Pipeline(lang='french', embedding='xlm-roberta-large', gpu=True)
    elif method['fr'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['fr'] = spacy.load("fr_core_news_sm")

if need_load_model_de:
    if method['de'] == 'Stanza':
        nlp_token['de'] = stanza.Pipeline(lang="de", package='gsd', processors='tokenize')
    elif method['de'] == 'Trankit':
        nlp_token['de'] = trankit.Pipeline(lang='german', embedding='xlm-roberta-large', gpu=True)
    elif method['de'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['de'] = spacy.load("de_core_news_sm")

if need_load_model_uk:
    if method['uk'] == 'Stanza':
        nlp_token['uk'] = stanza.Pipeline(lang="uk", package='iu', processors='tokenize')
    elif method['uk'] == 'Trankit':
        nlp_token['uk'] = trankit.Pipeline(lang='ukrainian', embedding='xlm-roberta-large', gpu=True)
    elif method['uk'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['uk'] = spacy.load("uk_core_news_sm")

if need_load_model_tr:
    if method['tr'] == 'Stanza':
        nlp_token['tr'] = stanza.Pipeline(lang="tr", package='imst', processors='tokenize')
    elif method['tr'] == 'Trankit':
        nlp_token['tr'] = trankit.Pipeline(lang='turkish', embedding='xlm-roberta-large', gpu=True)
    elif method['tr'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['tr'] = spacy.load("xx_ent_wiki_sm")
        nlp_token['tr'].add_pipe('sentencizer')

if need_load_model_ar:
    if method['ar'] == 'Stanza':
        nlp_token['ar'] = stanza.Pipeline(lang="ar", package='padt', processors='tokenize')
    elif method['ar'] == 'Trankit':
        nlp_token['ar'] = trankit.Pipeline(lang='arabic', embedding='xlm-roberta-large', gpu=True)
    elif method['ar'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['ar'] = spacy.load("xx_ent_wiki_sm")
        nlp_token['ar'].add_pipe('sentencizer')

# --------------------------------------------------


def output_sentences_stanza(text, lang):
    sentences = []
    doc = nlp_token[lang](text)
    for sentence in doc.sentences:
        sentences.append(sentence.text)
    return sentences


def output_sentences_spacy(text, lang):
    sentences = []
    doc = nlp_token[lang](text)
    for sentence in doc.sents:
        sentences.append(sentence.text)
    return sentences


def output_sentences_trankit(text, lang):
    sentences = []
    sents = nlp_token[lang].ssplit(text)
    for sent in sents['sentences']:
        sentences.append(sent['text'])
    return sentences


def output_sentences_natasha(text):
    sentences = []
    doc = Doc(text)
    doc.segment(segmenter)
    for sent in doc.sents:
        sentences.append(sent.text)
    return sentences


def output_sentences_nltk(text, lang):
    langs = {
        'en': 'english',
        'ru': 'russian',
        'es': 'spanish',
        'fr': 'french',
        'de': 'german',
        'uk': 'russian',
        'tr': 'turkish',
        'ar': 'english'
    }
    return sent_tokenize(text, language=langs[lang])


def start_position(sentence, text):
    symbols = 1
    try:
        while len(text.split(sentence[:symbols])) > 2 and symbols < len(sentence):
            symbols += 1
    except:
        while len(text.split()) > 2 and symbols < len(sentence):
            symbols += 1
    return text.find(sentence[:symbols])


def correct_sentences_ru(sentences, positions, text):
    sentences_corrected = []
    pos_corrected = []

    for i in range(len(sentences)):
        if len(sentences_corrected) > 0:
            last_sent = sentences_corrected[-1]
            last_pos = pos_corrected[-1]
            pos_ending_symbol = last_pos + len(last_sent)
            # Correct of incorrect splitting "Газета.Ru"
            if sentences[i].startswith('Ru\"'):
                if text[last_pos:last_pos+len(last_sent)+len(sentences[i])] == last_sent + sentences[i]:
                    sentences_corrected[-1] += sentences[i]
                elif text[last_pos:last_pos+len(last_sent)+len(sentences[i]) + 1] == last_sent + text[pos_ending_symbol] + sentences[i]:
                    sentences_corrected[-1] += text[pos_ending_symbol] + sentences[i]
            # Merge sentences if the previous sentence ends with a character from the set ['\"', ',']
            elif len(last_sent) > 1 and last_sent[-1] in ['\"', ','] \
                    and last_sent[-2] not in string.punctuation and text[pos_ending_symbol] != '\n' and \
                    text[last_pos:last_pos+len(last_sent)+len(sentences[i]) + 1] == last_sent + text[pos_ending_symbol] + sentences[i]:
                sentences_corrected[-1] += text[pos_ending_symbol] + sentences[i]
            else:
                sentences_corrected.append(sentences[i])
                pos_corrected.append(positions[i])
        else:
            sentences_corrected.append(sentences[i])
            pos_corrected.append(positions[i])

    return sentences_corrected, pos_corrected


def correct_sentences_all_langs(sentences, positions, text):
    sentences_corrected = []
    pos_corrected = []

    for i in range(len(sentences)):
        if len(sentences_corrected) > 0:
            last_sent = sentences_corrected[-1]
            last_pos = pos_corrected[-1]
            # Merge sentences if the previous sentence consists of one words
            if len(regexp_tokenize(last_sent, r'\w+')) == 1 and last_sent[-1] not in string.punctuation and \
                    text[last_pos:last_pos+len(last_sent)+len(sentences[i]) + 1] == last_sent + ' ' + sentences[i]:
                sentences_corrected[-1] += ' ' + sentences[i]
            else:
                sentences_corrected.append(sentences[i])
                pos_corrected.append(positions[i])
        else:
            sentences_corrected.append(sentences[i])
            pos_corrected.append(positions[i])

    return sentences_corrected, pos_corrected


def preprocess(text, lang, min_sent_len):
    sentences = []
    if text and lang is not None:
        text_splitted = []
        if method[lang] == 'Stanza':
            text_splitted = output_sentences_stanza(text, lang)
        elif method[lang] == 'Trankit':
            text_splitted = output_sentences_trankit(text, lang)
        elif method[lang] == 'Spacy':
            text_splitted = output_sentences_spacy(text, lang)
        elif method[lang] == 'NLTK':
            text_splitted = output_sentences_nltk(text, lang)
        elif method[lang] == 'Natasha' and lang == 'ru':
            text_splitted = output_sentences_natasha(text)
        for sentence in text_splitted:
            if sentence.strip() != '':
                sentences.append(sentence.strip())

    sentences_temp = []
    for sentence in sentences:
        sentences_temp.extend(sentence.split('\n'))
    sentences = sentences_temp

    positions = []
    pos = 0
    for sentence in sentences:
        pos += start_position(sentence, text[pos:])
        positions.append(pos)
        pos += len(sentence)

    sentences, positions = correct_sentences_all_langs(sentences, positions, text)

    if lang == "ru":
        sentences, positions = correct_sentences_ru(sentences, positions, text)

    sentences_filtered = []
    positions_filtered = []
    for sentence, start in zip(sentences, positions):
        if len(regexp_tokenize(sentence, r'\w+')) >= min_sent_len:
            sentences_filtered.append(sentence)
            positions_filtered.append(start)

    return sentences_filtered, positions_filtered


def get_sentences(text, lang, min_sent_len):
    sentences_dict = []
    sentences, positions = preprocess(text, lang, min_sent_len)
    for sentence, start in zip(sentences, positions):
        sentences_dict.append({'text': sentence, 'start': start})

    return sentences_dict


def segmentation(messages, min_sent_len=3):
    """ Функция сегментации входных сообщений на предложения.
    
    :param messages: список входных сообщений в формате JSON (dict). Каждое сообщение содержит поле `lang` - язык.
    :param min_sent_len: минимальное количество слов в предложении, при котором оно сохраняется в поле `sentences`.
    
    :return:
        список входных сообщений, для каждого из которых добавлено поле `sentences`: список словарей; каждый словарь = предложение; словарь включает поля:
            - `start` - позиция начала предложения во входном сообщении;
            - `text` - текст предложения.
    
    """

    messages_new = []

    for i in range(len(messages)):
        message = copy.deepcopy(messages[i])

        message_id = ''
        if 'id' in message.keys():
            message_id = message['id']
        
        if 'lang' in message and 'text' in message:
            try:
                sentences = get_sentences(message['text'], message['lang'], min_sent_len)
                if len(sentences) > 0:
                    message['sentences'] = sentences
                    messages_new.append(message)
            except:
                e = sys.exc_info()[1]
                t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print("%s Text ID: %s, Error: %s" % (t, message_id, e))

    return messages_new
