from datetime import datetime
import regex as re
import stanza
import copy
import trankit
import spacy
from natasha import (
    Segmenter,
    Doc
)
from nltk.tokenize import word_tokenize
import sys

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

nlp_token = {}
model = {
    'en': 'Stanza',
    'ru': 'Stanza',
    'es': 'Stanza',
    'fr': 'Stanza',
    'de': 'Stanza',
    'uk': 'Stanza',
    'tr': 'Stanza',
    'ar': 'Stanza',
}

if need_load_model_en:
    if model['en'] == 'Stanza':
        nlp_token['en'] = stanza.Pipeline(lang="en", package='ewt', processors='tokenize')
    elif model['en'] == 'Trankit':
        nlp_token['en'] = trankit.Pipeline(lang="english", gpu=True)
    elif model['en'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['en'] = spacy.load("en_core_web_sm")

if need_load_model_ru:
    if model['ru'] == 'Stanza':
        nlp_token['ru'] = stanza.Pipeline(lang="ru", package='gsd', processors='tokenize')
    elif model['ru'] == 'Trankit':
        nlp_token['ru'] = trankit.Pipeline(lang="russian", gpu=True)
    elif model['ru'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['ru'] = spacy.load("ru_core_news_sm")
    elif model['ru'] == 'Natasha':
        segmenter = Segmenter()

if need_load_model_es:
    if model['es'] == 'Stanza':
        nlp_token['es'] = stanza.Pipeline(lang="es", package='gsd', processors='tokenize')
    elif model['es'] == 'Trankit':
        nlp_token['es'] = trankit.Pipeline(lang="spanish", gpu=True)
    elif model['es'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['es'] = spacy.load("es_core_news_sm")

if need_load_model_fr:
    if model['fr'] == 'Stanza':
        nlp_token['fr'] = stanza.Pipeline(lang="fr", package='gsd', processors='tokenize')
    elif model['fr'] == 'Trankit':
        nlp_token['fr'] = trankit.Pipeline(lang="french", gpu=True)
    elif model['fr'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['fr'] = spacy.load("fr_core_news_sm")

if need_load_model_de:
    if model['de'] == 'Stanza':
        nlp_token['de'] = stanza.Pipeline(lang="de", package='gsd', processors='tokenize')
    elif model['de'] == 'Trankit':
        nlp_token['de'] = trankit.Pipeline(lang="german", gpu=True)
    elif model['de'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['de'] = spacy.load("de_core_news_sm")

if need_load_model_uk:
    if model['uk'] == 'Stanza':
        nlp_token['uk'] = stanza.Pipeline(lang="uk", package='iu', processors='tokenize')
    elif model['uk'] == 'Trankit':
        nlp_token['uk'] = trankit.Pipeline(lang="ukrainian", gpu=True)
    elif model['uk'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['uk'] = spacy.load("uk_core_news_sm")

if need_load_model_tr:
    if model['tr'] == 'Stanza':
        nlp_token['tr'] = stanza.Pipeline(lang="tr", package='imst', processors='tokenize')
    elif model['tr'] == 'Trankit':
        nlp_token['tr'] = trankit.Pipeline(lang="turkish", gpu=True)
    elif model['tr'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['tr'] = spacy.load("xx_ent_wiki_sm")

if need_load_model_ar:
    if model['ar'] == 'Stanza':
        nlp_token['ar'] = stanza.Pipeline(lang="ar", package='padt', processors='tokenize')
    elif model['ar'] == 'Trankit':
        nlp_token['ar'] = trankit.Pipeline(lang="arabic", gpu=True)
    elif model['ar'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_token['ar'] = spacy.load("xx_ent_wiki_sm")

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


def get_tokens_stanza(text, lang):
    doc = nlp_token[lang](text)
    tokens_list = []
    for sentence in doc.sentences:
        for token in sentence.tokens:
            tokens_list = append_token(token.text, tokens_list)
    return tokens_list


def get_tokens_trankit(text, lang):
    doc = nlp_token[lang].tokenize(text, is_sent=True)
    tokens_list = []
    for token in doc['tokens']:
        tokens_list = append_token(token['text'], tokens_list)
    return tokens_list


def get_tokens_spacy(text, lang):
    doc = nlp_token[lang](text)
    tokens_list = []
    for token in doc:
        tokens_list = append_token(token.text, tokens_list)
    return tokens_list


def get_tokens_nltk(sent_text, lang):
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
    tokens_list = []
    for token in word_tokenize(sent_text, language=langs[lang]):
        tokens_list = append_token(token, tokens_list)
    return tokens_list


def get_tokens_natasha(text):
    doc = Doc(text)
    tokens_list = []
    doc.segment(segmenter)
    for token in doc.tokens:
        tokens_list = append_token(token.text, tokens_list)
    return tokens_list


def join_tokens(text, grammar, lang):
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

    if lang == 'ru':
        letter_digit_complex_patterns.extend(['млн\.', 'тыс\.', 'руб\.', 'ст\.'])

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


def tokenization(messages):
    """ Функция токенизации входных сообщений.

    Токенизируются предложения (заранее выделенные) и заголовок сообщения.
    
    :param messages: список входных сообщений в формате JSON (dict). Каждое сообщение содержит поля:
            - `lang` - язык;
            - `sentences` - список предложений. Каждое предложение представляет собой словарь, содержащий поле `text`;
            - `title` - заголовок сообщения (опционально).

    :return:
        - список входных сообщений, для каждого из которых выполнено следующее:
            - для каждого предложения поля `sentences` добавлено поле `grammar` - список слов, в котором будут в дальнейшем содержаться результаты морфологического и синтаксического анализа. Это список списков; каждый вложенный список включает 3 элемента:
                - исходная словоформа (регистрозависимая) - функция записывает сюда выделенную (токенизированную) словоформу;
                - нормальная форма слова, приведенная к нижнему регистру - функция записывает сюда пустую строку;
                - часть речи - функция записывает сюда пустую строку;
                - время глагола (в случае отсутствия - пустая строка) - функция записывает сюда пустую строку;
                - ID данного слова в предложении, начиная с единицы - функция записывает сюда -1;
                - ID слова, с которым связано данное слово - функция записывает сюда -1;
            - добавлено поле `title_analysis` (словарь), в котором добавлено поле `grammar` - аналогичное по смыслу.
        
    """

    messages_new = copy.deepcopy(messages)

    for message in messages_new:
        message_id = ''
        if 'id' in message.keys():
            message_id = message['id']
        
        if 'lang' in message:
            if 'sentences' in message and len(message['sentences']) > 0:
                for sentence in message['sentences']:
                    try:
                        text = sentence['text']
                        if model[message['lang']] == 'Stanza':
                            sentence['grammar'] = get_tokens_stanza(text, message['lang'])
                        elif model[message['lang']] == 'Trankit':
                            sentence['grammar'] = get_tokens_trankit(text, message['lang'])
                        elif model[message['lang']] == 'Natasha' and message['lang'] == 'ru':
                            sentence['grammar'] = get_tokens_natasha(text)
                        elif model[message['lang']] == 'Spacy':
                            sentence['grammar'] = get_tokens_spacy(text, message['lang'])
                        elif model[message['lang']] == 'NLTK':
                            sentence['grammar'] = get_tokens_nltk(text, message['lang'])
                        if 'grammar' in sentence:
                            sentence['grammar'] = join_tokens(text, sentence['grammar'], message['lang'])
                    except:
                        e = sys.exc_info()[1]
                        t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                        print("%s Module: %s, function: %s, text ID: %s, error: %s" % (t, 'tokenization', 'tokenization', message_id, e))

            if 'title' in message and len(message['title']) > 0:
                try:
                    if model[message['lang']] == 'Stanza':
                        message['title_analysis'] = {'grammar': get_tokens_stanza(message['title'], message['lang'])}
                    elif model[message['lang']] == 'Trankit':
                        message['title_analysis'] = {'grammar': get_tokens_trankit(message['title'], message['lang'])}
                    elif model[message['lang']] == 'Natasha' and message['lang'] == 'ru':
                        message['title_analysis'] = {'grammar': get_tokens_natasha(message['title'])}
                    elif model[message['lang']] == 'Spacy':
                        message['title_analysis'] = {'grammar': get_tokens_spacy(message['title'], message['lang'])}
                    elif model[message['lang']] == 'NLTK':
                        message['title_analysis'] = {'grammar': get_tokens_nltk(message['title'], message['lang'])}
                    if 'grammar' in message['title_analysis']:
                        message['title_analysis']['grammar'] = join_tokens(message['title'],
                                                                           message['title_analysis']['grammar'],
                                                                           message['lang'])
                except:
                    e = sys.exc_info()[1]
                    t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    print("%s Module: %s, function: %s, text ID: %s, error: %s" % (t, 'tokenization', 'tokenization', message_id, e))
    return messages_new
