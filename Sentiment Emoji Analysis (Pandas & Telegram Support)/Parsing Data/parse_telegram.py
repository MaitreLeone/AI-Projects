import requests
import json
from bs4 import BeautifulSoup

with open('page_readovka.html', 'r', encoding='utf-8') as f:
    file = f.readlines()

file = ' '.join(file)
soup = BeautifulSoup(file, 'html.parser')
print(soup)
posts = soup.find_all('div', class_='card card-body border p-2 px-1 px-sm-3 post-container')
for answer in posts:
    posts = soup.find_all('div', class_="post-text")
    post_list = [answer.text.strip() for answer in posts if answer.text != ""]
    watches = soup.find_all('a', class_='btn btn-light btn-rounded py-05 px-13 mr-1 popup_ajax font-12 font-sm-13')
    watches_list = []
    reactions_list = []
    for watch in watches:
        for k, v in watch.attrs.items():
            if k == 'data-original-title' and v == 'Количество просмотров публикации':
                watch_str = watch.text.replace('\n', '')
                watch_str = watch_str.strip()
                watches_list.append(watch_str)
    reactions = soup.find_all('span', class_='btn btn-light btn-rounded py-05 px-13 mr-1 font-12 font-sm-13')
    for reaction in reactions:
        for k, v in reaction.attrs.items():
            if k == 'data-original-title' and 'Количество реакций к публикации' in v:
                #reactions_list.append(v)
                reactions = []
                reaction_codes = ['\U0001F44D', '\U0001F44E', '\U0001F601', '\U0001F631', '\U0001F525', '\U0001F622', '\U00002764', '\U0001F92C', '\U0001F44F']
                for i in range(len(v.split())):
                    for code in reaction_codes:
                        if code in v.split()[i]:
                            reaction = code + '-' + v.split()[i+2][0:]
                            reaction = reaction.replace('</div><div', '')
                            if '</div>' in reaction:
                                reaction = reaction.replace('</div>', '')
                            reactions.append(reaction)
                reactions_list.append(reactions)
print(post_list)
print(len(post_list))
print(watches_list)
print(len(watches_list))
print(reactions_list)
print(len(reactions_list))
json_list = []
for reaction in reactions_list:
    json_list.append({
        'text': post_list[reactions_list.index(reaction)],
        'watched': watches_list[reactions_list.index(reaction)],
        'reactions': reaction,
    })
with open('readovka.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(json_list, ensure_ascii=False, indent=2))
'''    
answers = soup.find_all('a', class_='btn btn-light btn-rounded py-05 px-13 mr-1 popup_ajax font-12 font-sm-13')
for answer in answers:
    print(answer.text)
answers = soup.find_all('span', class_='btn btn-light btn-rounded py-05 px-13 mr-1 font-12 font-sm-13')
for answer in answers:
    print(answer.text)
'''