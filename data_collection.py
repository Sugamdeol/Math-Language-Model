import requests

def get_wikipedia_article(title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=extracts&format=json&exlimit=max&explaintext=1"
    response = requests.get(url)
    pages = response.json()['query']['pages']
    page = next(iter(pages.values()))
    return page['extract']

math_articles = ["Mathematics", "Calculus", "Linear algebra", "Probability", "Statistics"]
articles_text = [get_wikipedia_article(title) for title in math_articles]

with open('math_articles.txt', 'w', encoding='utf-8') as file:
    for article in articles_text:
        file.write(article + '\n\n')
