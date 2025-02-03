import pandas as pd
import json
filename = 'cut_the_crap.json'
with open(filename, 'r', encoding='utf-8') as f:
    input_messages = json.load(f)

texts = []
watches = []
emotions = []
reaction_codes = ['\U0001F44D', '\U0001F44E', '\U0001F601', '\U0001F631', '\U0001F525', '\U0001F622', '\U00002764', '\U0001F92C', '\U0001F44F']
for r in reaction_codes:
    emotions.append([0 for i in range(len(input_messages))])
for r in reaction_codes:
    print(r)

for message in input_messages:
    texts.append(message['text'])
    watches.append(message['watched'])
    for reaction in message['reactions']:
        reaction = reaction.split('-')
        if reaction[0] in reaction_codes:
            emotions[reaction_codes.index(reaction[0])][input_messages.index(message)] += int(reaction[1])

json_list = []
json_list.append({
    'text': texts,
    'watched': watches,
})
for elem in json_list:
    for r in reaction_codes:
        elem[r] = emotions[reaction_codes.index(r)]
with open('cut_the_crap_new.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(json_list, ensure_ascii=False, indent=2))




        


