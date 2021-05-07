import requests
from bs4 import BeautifulSoup

# 서울대 공대 교수 이름 따오기 크롤링

for i in range(1,64):
    url = 'https://eng.snu.ac.kr/professor?&title=&page='+str(i)

    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.select_one('#block-system-main > div > div > div.view-content')

    names = content.select('dt')
    departments = content.select('dd')

    for name in names:
        print(name.get_text())


#    for depart in departments:
#        temp = depart.text.split('\n')
#        print(temp[1].split()[2])


#    for email in departments:
#        temp = email.text.split('\n')
#        print(temp[3].split()[2])


