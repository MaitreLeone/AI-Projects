from datetime import datetime
from navec import Navec
from slovnet import Syntax
from spacy.tokens import Doc as spacy_doc
import stanza
import trankit
import spacy
import copy
import sys
import os

# ---------------- SETTINGS SECTION ----------------

# Path to the directory with models for Natasha
model_path = 'models'

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

nlp_syntax = {}

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
        nlp_syntax['en'] = stanza.Pipeline(lang="en", package='ewt', processors='tokenize,mwt,pos,lemma,depparse',
                                           tokenize_pretokenized=True)
    elif method['en'] == 'Trankit':
        nlp_syntax['en'] = trankit.Pipeline(lang='english')
    elif method['en'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_syntax['en'] = spacy.load("en_core_web_sm")

if need_load_model_ru:
    if method['ru'] == 'Stanza':
        nlp_syntax['ru'] = stanza.Pipeline(lang="ru", package='syntagrus', processors='tokenize,pos,lemma,depparse',
                                           tokenize_pretokenized=True)
    elif method['ru'] == 'Trankit':
        nlp_syntax['ru'] = trankit.Pipeline(lang='russian')
    elif method['ru'] == 'Natasha':
        navec = Navec.load(os.path.join(model_path, 'navec_news_v1_1B_250K_300d_100q.tar'))
        nlp_syntax_natasha = Syntax.load(os.path.join(model_path, 'slovnet_syntax_news_v1.tar'))
        nlp_syntax_natasha.navec(navec)

if need_load_model_es:
    if method['es'] == 'Stanza':
        nlp_syntax['es'] = stanza.Pipeline(lang="es", package='gsd', processors='tokenize,mwt,pos,lemma,depparse',
                                           tokenize_pretokenized=True)
    elif method['es'] == 'Trankit':
        nlp_syntax['es'] = trankit.Pipeline(lang='spanish')
    elif method['es'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_syntax['es'] = spacy.load("es_core_news_sm")

if need_load_model_fr:
    if method['fr'] == 'Stanza':
        nlp_syntax['fr'] = stanza.Pipeline(lang="fr", package='gsd', processors='tokenize,mwt,pos,lemma,depparse',
                                           tokenize_pretokenized=True)
    elif method['fr'] == 'Trankit':
        nlp_syntax['fr'] = trankit.Pipeline(lang='french')
    elif method['fr'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_syntax['fr'] = spacy.load("fr_core_news_sm")

if need_load_model_de:
    if method['de'] == 'Stanza':
        nlp_syntax['de'] = stanza.Pipeline(lang="de", package='gsd', processors='tokenize,mwt,pos,lemma,depparse',
                                           tokenize_pretokenized=True)
    elif method['de'] == 'Trankit':
        nlp_syntax['de'] = trankit.Pipeline(lang='german')
    elif method['de'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_syntax['de'] = spacy.load("de_core_news_sm")

if need_load_model_uk:
    if method['uk'] == 'Stanza':
        nlp_syntax['uk'] = stanza.Pipeline(lang="uk", package='iu', processors='tokenize,mwt,pos,lemma,depparse',
                                           tokenize_pretokenized=True)
    elif method['uk'] == 'Trankit':
        nlp_syntax['uk'] = trankit.Pipeline(lang='ukrainian')
    elif method['uk'] == 'Spacy':
        spacy.prefer_gpu()
        nlp_syntax['de'] = spacy.load("uk_core_news_sm")

if need_load_model_tr:
    if method['tr'] == 'Stanza':
        nlp_syntax['tr'] = stanza.Pipeline(lang="tr", package='imst', processors='tokenize,mwt,pos,lemma,depparse',
                                           tokenize_pretokenized=True)
    elif method['tr'] == 'Trankit':
        nlp_syntax['tr'] = trankit.Pipeline(lang='turkish')

if need_load_model_ar:
    if method['ar'] == 'Stanza':
        nlp_syntax['ar'] = stanza.Pipeline(lang="ar", package='padt', processors='tokenize,mwt,pos,lemma,depparse',
                                           tokenize_pretokenized=True)
    elif method['ar'] == 'Trankit':
        nlp_syntax['ar'] = trankit.Pipeline(lang='arabic')

# --------------------------------------------------


def get_syntax_stanza(grammar, lang):
    tokens = []
    for i in range(len(grammar)): 
        tokens.append([token[0] for token in grammar[i]])
    
    doc = nlp_syntax[lang](tokens)  # let pretokeinzed text to model
    
    for i in range(len(grammar)):
        word_ids = []
        head_ids = []
        
        for word in doc.sentences[i].words:
            word_ids.append(word.id)
            head_ids.append(word.head)

        for j in range(len(grammar[i])):
            grammar[i][j][4] = word_ids[j]
            grammar[i][j][5] = head_ids[j]

    return grammar


def get_syntax_trankit(grammar, lang):
    tokens = []
    for i in range(len(grammar)): 
        tokens.append([token[0] for token in grammar[i]])
    
    doc = nlp_syntax[lang].posdep(tokens)  # let pretokeinzed text to model
    
    for i in range(len(grammar)):
        word_ids = []
        head_ids = []
        for word in doc['sentences'][i]['tokens']:
            word_ids.append(word['id'])
            head_ids.append(word['head'])

        for j in range(len(grammar[i])):
            grammar[i][j][4] = word_ids[j]
            grammar[i][j][5] = head_ids[j]

    return grammar


def get_syntax_spacy(grammar, lang):
    tokens = []
    for i in range(len(grammar)): 
        tokens.append([token[0] for token in grammar[i]])

    docs = [spacy_doc(nlp_syntax[lang].vocab, tokens[i]) for i in range(len(tokens))]
    docs = nlp_syntax[lang].pipe(docs) # let pretokeinzed text to model
    
    for i, doc in enumerate(docs):
        word_ids = []
        head_ids = []
        for token in doc:
            word_ids.append(token.i + 1)
            if token.i != token.head.i:
                head_ids.append(token.head.i + 1)
            else:
                head_ids.append(0)

        for j in range(len(grammar[i])):
            grammar[i][j][4] = word_ids[j]
            grammar[i][j][5] = head_ids[j]

    return grammar


def get_syntax_natasha(grammar):
    for i in range(len(grammar)):
        words = [token[0] for token in grammar[i]]
        
        markup_syntax = next(nlp_syntax_natasha.map([words]))
        
        word_ids = []
        head_ids = []
        for token in markup_syntax.tokens:
            word_ids.append(int(token.id))
            head_ids.append(int(token.head_id))

        for j in range(len(grammar[i])):
            grammar[i][j][4] = word_ids[j]
            grammar[i][j][5] = head_ids[j]

    return grammar


def syntax_analysis(messages):
    """ Функция синтаксического анализа входных сообщений.
    
    Синтаксический анализ осуществляется для отдельных предложений (поле `sentences`) и для заголовка (поля `title` и `title_analysis`).
    
    На вход функции предложения и заголовок приходят токенизированными (есть поля `grammar` с результатами морфологического анализа).
    
    :param messages: список входных сообщений в формате JSON (dict). Каждое сообщение содержит поля:
            - `lang` - язык;
            - `sentences` - список предложений. Каждое предложение представляет собой словарь, содержащий поля:
                - `text` - исходный текст предложения;
                - `grammar` - список слов (списков), для каждого слова имеются результаты морфологического анализа, в том числе, нормальная форма и часть речи;
                - `ner` - перечень именованных сущностей;
            - `title` - заголовок сообщения (опционально).
            - `title_analysis` - результаты анализа заголовка (опционально) - словарь, в котором имеются поля:
                - `grammar` - аналогичное по смыслу полю `grammar` для предложений;
                - `ner` - аналогичное по смыслу полю `ner` для предложений.

    :return:
        список входных сообщений, для каждого из которых:
            - для каждого предложения (в поле `sentences`) в поле `grammar` добавлены результаты синтаксического анализа. Это список списков; каждый вложенный список включает 6 элементов:
                - исходная словоформа (регистрозависимая) - не меняется функцией;
                - нормальная форма слова, приведенная к нижнему регистру - не меняется функцией;
                - часть речи - не меняется функцией;
                - время глагола (в случае отсутствия - пустая строка) - заполняется функцией;
                - ID данного слова в предложении, начиная с единицы - заполняется функцией;
                - ID слова, с которым связано данное слово - заполняется функцией;
            - в результаты анализа заголовка (поле `title_analysis`) в поле `grammar` добавлены результаты синтаксического анализа - аналогичные по смыслу полю `grammar` для предложений;
    
    """

    messages_new = copy.deepcopy(messages)

    for message in messages_new:
        message_id = ''
        if 'id' in message.keys():
            message_id = message['id']
        
        if 'lang' in message:
            lang = message['lang']
            
            if 'sentences' in message and len(message['sentences']) > 0:
                try:
                    sentences = message['sentences']
                    grammar = [sentence['grammar'] for sentence in sentences]
                    if method[lang] == 'Stanza':
                        grammar = get_syntax_stanza(grammar, lang)
                    elif method[lang] == 'Trankit':
                        grammar = get_syntax_trankit(grammar, lang)
                    elif method[lang] == 'Spacy':
                        grammar = get_syntax_spacy(grammar, lang)
                    elif method[lang] == 'Natasha' and lang == 'ru':
                        grammar = get_syntax_natasha(grammar)
                    for i in range(len(sentences)):
                        sentences[i]['grammar'] = grammar[i]
                except:
                    e = sys.exc_info()[1]
                    t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    print("%s Module: %s, function: %s, text ID: %s, error: %s" % (t, 'syntax', 'syntax_analysis', message_id, e))

            if 'title_analysis' in message:
                try:
                    title_analysis = message['title_analysis']
                    grammar = [title_analysis['grammar']]
                    if method[lang] == 'Stanza':
                        grammar = get_syntax_stanza(grammar, lang)
                    elif method[lang] == 'Trankit':
                        grammar = get_syntax_trankit(grammar, lang)
                    elif method[lang] == 'Spacy':
                        grammar = get_syntax_spacy(grammar, lang)
                    elif method[lang] == 'Natasha' and lang == 'ru':
                        grammar = get_syntax_natasha(grammar)
                    title_analysis['grammar'] = grammar[0]
                except:
                    e = sys.exc_info()[1]
                    t = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    print("%s Module: %s, function: %s, text ID: %s, error: %s" % (t, 'syntax', 'syntax_analysis', message_id, e))

    return messages_new
