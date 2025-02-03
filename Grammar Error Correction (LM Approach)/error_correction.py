from transformers import AutoModelForMaskedLM, AutoTokenizer
from silero import silero_te
import language_tool_python
import pymorphy2
import string
import torch
import copy
import requests
import itertools
import os
from wrappers import WrapperSegm, WrapperMorph, WrapperToken, WrapperSyntax

import torch.nn.functional as F
from nltk.stem.snowball import SnowballStemmer

# ---------------- SETTINGS SECTION ----------------

# Number of attempts for correcting ending of the word
attempts_change_ending = 50000

#proba_gce_treshold = 0.01

# Directory with models
# 172.16.211.106
models_dir = '/opt/models'

# LanguageTool
host = 'http://172.16.211.111:8081'

# list of languages to load
languages = ['ru']  # ['ru', 'en', 'es', 'de', 'fr', 'ar', 'tr', 'uk']


# --------------- LOAD MODELS ----------------

ner_models = {}

porter_stem = None
silero_model = None

if 'ru' in languages:
    cases = ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct']
    numbers = ['sing', 'plur']
    genders = ['masc', 'femn', 'neut']
    tenses = ['past', 'pres', 'futr']

def init():
    global languages, porter_stem, silero_model, tokenizer_bert, model_bert,\
    PRE_TRAINED_MODEL_NAME_BERT, wr_segm, wr_token, wr_morph, wr_syntax
    
    wr_segm = WrapperSegm()
    wr_token = WrapperToken()
    wr_morph = WrapperMorph(with_ner=False)
    wr_syntax = WrapperSyntax()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if 'ru' in languages:
        PRE_TRAINED_MODEL_NAME_BERT = os.path.join(models_dir, 'ruBert-large')
        tokenizer_bert = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME_BERT)
        model_bert = AutoModelForMaskedLM.from_pretrained(PRE_TRAINED_MODEL_NAME_BERT)
        model_bert.eval()
        model_bert.to(device)
        porter_stem = SnowballStemmer("russian")

        silero_model, example_texts, languages, punct, apply_te = silero_te()
# --------------------------------------------------

# ------------- WORD ENDING CORRECTION -------------
def get_word(p, param):
    if len(param) == 3:
        word = p.inflect({param[0], param[1], param[2]})
    else:
        word = p.inflect({param[0], param[1]})
    if word is not None:
        return word.word
    else:
        return None


def get_inflected_words(keyword):
    morph = pymorphy2.MorphAnalyzer()
    p = morph.parse(keyword)[0]
    inflected_words = []
    if 'VERB' in p.tag or 'INFN' in p.tag:
        for param in itertools.product(numbers, tenses, genders):
            word = get_word(p, param)
            if word is not None:
                inflected_words.append(word)
        for param in itertools.product(numbers, tenses):
            word = get_word(p, param)
            if word is not None:
                inflected_words.append(word)
        for param in itertools.product(tenses, genders):
            word = get_word(p, param)
            if word is not None:
                inflected_words.append(word)
    else:
        for param in itertools.product(numbers, cases, genders):
            word = get_word(p, param)
            if word is not None:
                inflected_words.append(word)
        for param in itertools.product(numbers, cases):
            word = get_word(p, param)
            if word is not None:
                inflected_words.append(word)
        for param in itertools.product(cases, genders):
            word = get_word(p, param)
            if word is not None:
                inflected_words.append(word)
    return list(set(inflected_words))


def correct_stem(stem, tokenizer_bert):
    bert_predicted_index = tokenizer_bert.encode(stem)[1:-1]
    roberta_tokens = []
    for index in bert_predicted_index:
        roberta_tokens.append(tokenizer_bert.decode([index]))
    if len(roberta_tokens[-1]) == 1:
        stem = tokenizer_bert.decode(bert_predicted_index[:-1])
    return stem


def check_match_lemmas(keyword_parsed, predicted_keyword_parsed):
    if predicted_keyword_parsed['lemma'] == keyword_parsed['lemma'] \
            and (('sing' in predicted_keyword_parsed['feats'] and 'sing' in keyword_parsed['feats'])
                 or ('plur' in predicted_keyword_parsed['feats'] and 'plur' in keyword_parsed['feats'])):
        return True
    else:
        return False

def check_match_feats(keyword, predicted_keyword, keyword_pos, predicted_keyword_pos, lang):
    messages_morph = wr_morph.do([{'text': keyword, 'lang': lang}])
    grammar = messages_morph[0]['sentences'][0]['grammar']
    keyword_parsed = {'lemma': grammar[0][0], 'feats': grammar[0][3]}
    messages_morph = wr_morph.do([{'text': predicted_keyword, 'lang': lang}])
    grammar = messages_morph[0]['sentences'][0]['grammar']
    predicted_keyword_parsed = {'lemma': grammar[0][0], 'feats': grammar[0][3]}
    if (keyword_pos == 'VERB' or keyword_pos == 'INFN') and keyword_pos == predicted_keyword_pos:
      if check_feats(keyword_parsed, predicted_keyword_parsed, numbers) or \
      check_feats(keyword_parsed, predicted_keyword_parsed, genders) or \
      check_feats(keyword_parsed, predicted_keyword_parsed, tenses):
        return True
      else:
        return False
    elif keyword_pos == predicted_keyword_pos:
      if check_feats(keyword_parsed, predicted_keyword_parsed, numbers) or \
      check_feats(keyword_parsed, predicted_keyword_parsed, genders):
        return True
      else:
        return False

def check_feats(keyword_parsed, predicted_keyword_parsed, feats):
  for feat in feats:
    if feat in predicted_keyword_parsed['feats'] and feat in keyword_parsed['feats']:
      return True
  return False

def get_corrected_keyword(keyword, keyword_stem, predicted_token, lang):
    punct = string.punctuation + '...'
    if predicted_token not in punct:
        messages_morph = wr_morph.do([{'text': keyword, 'lang': lang}])
        grammar = messages_morph[0]['sentences']['grammar']
        keyword_parsed = {'lemma': grammar[0][0], 'feats': grammar[0][3]}

        predicted_keyword = keyword_stem + predicted_token
        messages_morph = wr_morph.do([{'text': predicted_keyword, 'lang': lang}])
        grammar = messages_morph[0]['sentences']['grammar']
        predicted_keyword_parsed = {'lemma': grammar[0][0], 'feats': grammar[0][3]}

        if check_match_lemmas(keyword_parsed, predicted_keyword_parsed):
            return predicted_keyword

        predicted_keyword = keyword_stem[:-1] + predicted_token
        messages_morph = wr_morph.do([{'text': predicted_keyword, 'lang': lang}])
        grammar = messages_morph[0]['sentences']['grammar']
        predicted_keyword_parsed = {'lemma': grammar[0][0], 'feats': grammar[0][3]}
        if check_match_lemmas(keyword_parsed, predicted_keyword_parsed):
            return predicted_keyword

        predicted_keyword = porter_stem.stem(keyword) + predicted_token
        messages_morph = wr_morph.do([{'text': predicted_keyword, 'lang': lang}])
        grammar = messages_morph[0]['sentences']['grammar']
        predicted_keyword_parsed = {'lemma': grammar[0][0], 'feats': grammar[0][3]}
        if check_match_lemmas(keyword_parsed, predicted_keyword_parsed):
            return predicted_keyword

    return ''


def correct_keywords_col_bert(text, keywords_col, start_position, tokenizer_bert, model_bert, lang):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    messages_segm = wr_segm.do([{'text': text, 'lang': lang}])
    sentences = [sentence['text'] for sentence in messages_segm[0]['sentences']]
    predicted_keywords = []
    print('\t\tSentences:')
    for sentence in sentences:
        print('\t\t\t' + sentence)
    index_sentence = 0
    index_keyword_col = 0
    while index_sentence < len(sentences) and index_keyword_col < len(keywords_col):
        keyword_col = keywords_col[index_keyword_col]
        sentence = sentences[index_sentence]
        if keyword_col in sentence:
            print('\n\t\tSentence: ' + sentence)
            start_index_keyword_col = start_position
            prefix = sentence[:start_index_keyword_col].strip()
            tail = sentence[start_index_keyword_col + len(keyword_col):]
            
            messages_morph = wr_morph.do([{'text': keyword_col, 'lang': lang}])
            grammar = messages_morph[0]['sentences'][index_sentence]['grammar']
            keyword_col_token_pos = [[item[0], item[2]] for item in grammar]

            first_noun_index = 0
            while first_noun_index < len(keyword_col_token_pos) - 1 and keyword_col_token_pos[first_noun_index][1] != 'NOUN':
                first_noun_index += 1

            for j in reversed(range(first_noun_index + 1, len(keyword_col_token_pos))):
                tail = ' ' + keyword_col_token_pos[j][0] + tail
            for i in reversed(range(first_noun_index + 1)):
                keyword = keyword_col_token_pos[i][0]
                keyword_stem_nltk = porter_stem.stem(keyword)
                #keyword_stem = correct_stem(keyword_stem, tokenizer_bert)
                bert_predicted_indexes = tokenizer_bert.encode(keyword)[1:-1]
                token_list = []
                for index in bert_predicted_indexes:
                    token_list.append(tokenizer_bert.decode(index))
                if len(token_list) > 1:
                    last_token = token_list[-1].replace('##', '')
                    if len(token_list[-1]) > 1:
                        keyword_stem = ''.join(token_list[:-1])
                    elif len(token_list[-1]) == 1:
                        keyword_stem = ''
                elif len(token_list) == 1:
                    keyword_stem = ''
                else:
                    keyword_stem = None
                if '##' in keyword_stem:
                    keyword_stem = keyword_stem.replace('##', '')
                context = prefix + ''.join(
                    [' ' + x[0] for x in keyword_col_token_pos[:i]]) + ' ' + keyword_stem + tokenizer_bert.mask_token + tail

                print('\t\tContext: ' + context)

                indexed_tokens = tokenizer_bert.encode(context)
                tokens_tensor = torch.tensor([indexed_tokens])
                tokens_tensor = tokens_tensor.to(device)

                mask_token_index = torch.where(tokens_tensor == tokenizer_bert.mask_token_id)[1]

                with torch.no_grad():
                    outputs = model_bert(tokens_tensor)

                del tokens_tensor
                torch.cuda.empty_cache()

                logits = outputs.logits.to(device).cpu().detach()
                masked_token_logits = logits[0, mask_token_index.to(device).cpu().detach(), :]
                index_logit = [(k, masked_token_logits[0, k]) for k in range(len(masked_token_logits[0]))]
                index_logit.sort(key=lambda x: x[1], reverse=True)

                print('\t\tKeyword stem: ' + keyword_stem_nltk)
                k = 0
                found = False
                final_predicted_keyword = ''
                inflected_words = get_inflected_words(keyword)
                while k < attempts_change_ending:
                    predicted_token = tokenizer_bert.decode([index_logit[k][0]])
                    if predicted_token.startswith('##'):
                        predicted_token = predicted_token.replace('##', '')
                    punct = string.punctuation + '...'
                    if predicted_token not in punct and '[' not in predicted_token:
                        if keyword_stem == '':
                            predicted_keyword = predicted_token
                        else:
                            predicted_keyword = keyword_stem + predicted_token
                        predicted_keywords.append(predicted_keyword)
                        print('\t\t\t' + str(k + 1) + ' predicted token: ' + predicted_token +
                          ', predicted keyword: ' + predicted_keyword)
                    else:
                        predicted_keyword = ''

                    if len(predicted_keyword) > 0 and not found and len(predicted_keyword.split()) == 1:
                        if len(inflected_words) > 0:
                            morph = pymorphy2.MorphAnalyzer()
                            keyword_pos = morph.parse(keyword)[0].tag.POS
                            predicted_keyword_pos = morph.parse(predicted_keyword)[0].tag.POS
                            pos_compare = False
                            for word in inflected_words:
                                if keyword_pos == morph.parse(word)[0].tag.POS:
                                    pos_compare = True
                            if pos_compare and predicted_keyword.lower() in inflected_words \
                                and check_match_feats(keyword, predicted_keyword, keyword_pos, predicted_keyword_pos, lang):
                                final_predicted_keyword = predicted_keyword
                                found = True
                                break
                        else:
                            if not found and predicted_keyword.lower().startswith(keyword_stem_nltk):
                                final_predicted_keyword = predicted_keyword
                                found = True
                                break
                            if not found and predicted_keyword.lower().startswith(token_list[0]):
                                final_predicted_keyword = predicted_keyword
                                found = True
                                break
                    k += 1
                if not found and keyword not in tail:
                    tail = ' ' + keyword + tail
                elif found:
                    tail = ' ' + final_predicted_keyword + tail

            print('\t\tCorrected sentence: ' + prefix + tail)
            sentences[index_sentence] = prefix + tail
            sentences[index_sentence] = sentences[index_sentence].strip()
            index_keyword_col += 1
        else:
            start_index_search_kw = 0
            index_sentence += 1
    return ' '.join(sentence for sentence in sentences), predicted_keywords

# --------------------------------------------------

#исправление грамматических и орфографических ошибок с помощью LanguageTool (локально)
def grammar_correction(text, lang, replaces_dict=None):
    if replaces_dict != None:
        replaces = replaces_dict['new_string']
        replaces_indexes = [replaces_dict['result_start'], replaces_dict['result_start'] + len(replaces) - 1]
    if not lang:
        tool = language_tool_python.LanguageTool('auto', remote_server=host)
    else:
        tool = language_tool_python.LanguageTool(lang, remote_server=host)
    # print(tool.check('Ткст для прверки которя'))
    if replaces_dict == None:
        errors = tool.check(text)
        matches = []
        for error in errors:
            if error.replacements is not None and error.offset is not None:
                error_dict = {'error_word': text[error.offset:error.offset + error.errorLength],
                            'start': error.offset, 'end': error.offset + error.errorLength - 1,
                            'replacements': error.replacements}
                matches.append(error_dict)
        text_corrected = tool.correct(text)
        grammar_dict = {'corrected_text': text_corrected, 'number_errors': len(errors)}
        if len(errors) > 0:
            grammar_dict['errors'] = matches
    else:
        number_errors = 0
        text_corrected = text
        matches = []
        word = text[replaces_indexes[0]: replaces_indexes[1]]
        errors = tool.check(word)
        number_errors += len(errors)
        for error in errors:
            if error.replacements is not None and error.offset is not None:
                error_dict = {'error_word': text[error.offset:error.offset + error.errorLength],
                            'start': error.offset, 'end': error.offset + error.errorLength - 1,
                            'replacements': error.replacements}
                matches.append(error_dict)
        word_corrected = tool.correct(word)
        text_corrected[replaces_indexes[0]: replaces_indexes[1]] = word_corrected
        text_corrected = correct_sentence(text_corrected)
        grammar_dict = {'corrected_text': text_corrected, 'number_errors': len(errors)}
        if len(errors) > 0:
            grammar_dict['errors'] = matches
    return grammar_dict


#исправление пунктуционных ошибок в сегментированном тексте на предложения (предварительно очистив его от всех знаков пунктуации)
def punctuation(text, lang, replaces_dict=None):
    if replaces_dict != None:
        replaces = replaces_dict['new_string']
        replaces_indexes = replaces_indexes = [replaces_dict['result_start'], replaces_dict['result_start'] + len(replaces_dict['new_string'])]
        output_text = text
        word = text[replaces_indexes[0]: replaces_indexes[1]]
        if word[0].isupper():
            output_word = silero_model.enhance_text(word, lang)
        else:
            output_word = silero_model.enhance_text(word, lang).lower()
        if output_word != text.split()[-1]:
            signs = ['!', '?', '.']
            if output_word[-1] in signs:
                output_word = output_word.replace(output_word[-1], '')
        output_text = text[:replaces_indexes[0]] + output_word + text[replaces_indexes[1]:]
        return output_text
    else:
        sentences = get_sentences_without_punctuation(text, lang)
        new_sentences = []
        for sent in sentences:
            sent = sent.lower()
            output_text = silero_model.enhance_text(sent, lang)
            new_sentences.append(output_text)
        return correct_sentence(' '.join(new_sentences))


#удаление пробелов перед знаками препинания
def correct_sentence(sentence):
    for punct in string.punctuation:
        if punct in sentence:
            sentence = sentence.replace(' ' + punct, punct)
    return sentence


#очистка предложений от пунктуации
def get_sentences_without_punctuation(text, lang):
    sentences = []
    punct = string.punctuation
    messages_segm = wr_segm.do([{'text': text, 'lang': lang}])
    for sent in messages_segm[0]['sentences']:
        sentence = sent['text'].translate(str.maketrans('', '', punct))
        sentences.append(sentence)
    return sentences


def agreement(text, keywords, start_position, lang):
    corrected_sentence, predicted_keywords = correct_keywords_col_bert(text, keywords, start_position, tokenizer_bert, model_bert, lang)
    predicted_keywords = list(set(predicted_keywords))
    grammar_dict = {'corrected_text': corrected_sentence, 'replacements': [{'start': start_position, 'strings': predicted_keywords}]}
    return grammar_dict


def dep_word_agreement(original_text, changed_text, lang, replaces):
    #токенизация исходного и конечного предложений
    messages_original_tokens = wr_token.do([{'text': original_text, 'lang': lang}])
    messages_changed_tokens = wr_token.do([{'text': changed_text, 'lang': lang}])
    original_tokens = [item[0] for item in messages_original_tokens[0]['sentences'][0]['grammar']]
    changed_tokens = [item[0] for item in messages_changed_tokens[0]['sentences'][0]['grammar']]
    #получение списка изменённых слов в changed_text и их индексов
    #unique_changed_words = list(set(changed_tokens).difference(set(original_tokens)))
    unique_changed_words = []
    changed_words_indexes = []
    replaced_words = replaces['new_string']
    replaced_words_indexes = [replaces['result_start'], replaces['result_start'] + len(replaces['new_string']) - 1]
    for i in range(len(changed_tokens)):
        if changed_tokens[i] == replaced_words \
        and changed_tokens[i] == changed_text[replaced_words_indexes[0]: replaced_words_indexes[1] + 1]:
            unique_changed_words.append(changed_tokens[i])
            changed_words_indexes.append(i)
    #получение grammar на основе токенов исходного предложения для использования внешней функции без изменений
    original_tokens_grammar = [[token, '', '', '', '', ''] for token in original_tokens]
    #добавление в grammar значений параметров word_id и head_id из внешней функции синтаксического анализа
    
    messages_syntax = wr_syntax.do([{'text': original_text, 'lang': lang}])
    original_tokens_grammar = messages_syntax[0]['sentences'][0]['grammar']
    
    #получение списка head_ids (списка индексов связанных слов)
    head_ids = [item[5] for item in original_tokens_grammar]
    #получение списка связанных слов и списка их индексов в changed_text
    linked_words_list = []
    linked_words_indexes = []
    for index in changed_words_indexes:
        linked_words = []
        for i in range(len(head_ids)):
            if index + 1 in head_ids:
                if index + 1 == head_ids[i]:
                    linked_words.append(changed_tokens[i])
                    linked_words_indexes.append(i)
        linked_words_list.append(linked_words)
    #реализация морфологии для слов из unique_changed_words и связанных слов из linked_words
    morph = pymorphy2.MorphAnalyzer()
    morph_unique_list = []
    morph_linked_list = []
    for word in unique_changed_words:
        morph_unique_list.append(morph.parse(word)[0])
    for i in range(len(linked_words_list)):
        changed_words_list = []
        for elem in linked_words_list[i]:
            changed_words_list.append(morph.parse(elem)[0])
        morph_linked_list.append(changed_words_list)
    #получение списка морфологических признаков для слов из unique_changed_words
    morph_tags = [item.tag for item in morph_unique_list]
    #получение списка связанных слов с перенесёнными морфологическими признаками относительно признаков слов из unique_changed_words
    inflected_words_list = []
    for i, tag in enumerate(morph_tags):
        inflected_words = []
        for elem in morph_linked_list[i]:
            if 'VERB' in tag:
                inflected_words.append(elem.inflect({tag.number, tag.gender}).word)
            else:
                inflected_words.append(elem.inflect({tag.number, tag.gender, tag.case}).word)
        inflected_words_list.append(inflected_words)
    #замена форм связанных слов в конечном предложении
    for elem in inflected_words_list:
        linked_word_index = 0
        for i in range(len(changed_tokens)):
            if i in linked_words_indexes:
                changed_tokens[i] = elem[linked_word_index]
                linked_word_index += 1
    #возврат правильного предложения
    grammar_dict = {'corrected_text': correct_sentence(' '.join(changed_tokens))}
    return grammar_dict


# Parameter method can take one of the values "spelling", "punctuation", "grammar"
def error_correction(messages, methods=["spelling", 'punctuation', 'grammar']):
    messages_new = copy.deepcopy(messages)
    for message in messages_new:
        replaces = None
        lang = None
        if 'replaces' in message.keys():
            replaces = message['replaces']
        if 'lang' in message:
            lang = message['lang']
        grammar_dict = {}
        for method in methods:
            if method == 'spelling':
                if replaces == None:
                    text = message['text']
                    grammar_dict[method] = grammar_correction(text, lang)
                else:
                    for elem in replaces:
                        if elem['check']:
                            if 'text' in elem.keys() and 'result_text' in elem.keys():
                                grammar_dict[method] = grammar_correction(message['result_text'], lang, elem)
                            else:
                                grammar_dict[method] = grammar_correction(message['result_text'], lang)
            elif method == 'punctuation':
                if lang is not None:
                    if replaces == None:
                        text = message['text']
                        grammar_dict[method] = {'corrected_text': punctuation(text, lang)}
                    else:
                        for elem in replaces:
                            if elem['check']:
                                if 'text' in elem.keys() and 'result_text' in elem.keys():
                                    grammar_dict[method] = {'corrected_text': punctuation(message['result_text'], lang, elem)}
                                else:
                                    grammar_dict[method] = punctuation(message['result_text'], lang)
                else:
                    text = message['text']
                    grammar_dict[method] = {'corrected_text': text}
            elif method == 'grammar':
                if replaces != None:
                    for elem in replaces:
                        if elem['check']:
                            if 'text' in message.keys() and 'result_text' in message.keys() and 'old_string' in elem.keys():
                                grammar_dict[method] = dep_word_agreement(message['text'], message['result_text'], lang, elem)
                            else:
                                grammar_dict[method] = agreement(message['result_text'], [elem['new_string']], elem['result_start'], lang)
        if len(grammar_dict) > 0:
            message['error_correction'] = grammar_dict
    return messages_new

messages = [{
    "author_style": "Дмитрий Пучков",
    "lang": "ru",
    "result_text": "В интернетах ходят стойкие слухи, что продажей прокатных версий киноискусство в санкционное время занимается всего один барышник на рынке.",
    "replaces": [
      {
        "original_start": 49,
        "result_start": 64,
        "new_string": "киноискусство",
        "check": True
      },
      {
        "original_start": 96,
        "result_start": 120,
        "old_string": "торговец",
        "new_string": "барышник",
        "check": True
      }
    ]
  }]
init()
print(correct_keywords_col_bert(messages[0]['result_text'], ['киноискусство'], 64, tokenizer_bert, model_bert, 'ru'))