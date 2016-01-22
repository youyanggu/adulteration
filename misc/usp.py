import time

from bs4 import BeautifulSoup
import requests
import pandas as pd

def scrape_usp_data():
    URL = 'http://www.foodfraud.org/search/site?search_api_views_fulltext='
    COLUMNS = ['unique_id', 'regulatory_status', 'report_type', 'ingredient_category',
               'ingredient', 'adulterant', 'fraud_type', 'pub_year',
               'detection_method', 'ref', 'inline_ref', 'delete_node']

    START = 1880
    END = 2015
    years = range(START, END+1)

    data = {}
    for year in years:
        ret = requests.get(URL+str(year))
        soup = BeautifulSoup(ret.text, 'lxml')
        #tr_class = soup.find_all("tr", class_='views-row-first')
        tbody_class = soup.find_all('tbody')
        if len(tbody_class) == 0:
            print "Not found:", year
            continue
        else:
            assert len(tbody_class) == 1
            print "======================================================"
            print year
        tr_classes = tbody_class[0].find_all('tr')
        print "Entries:", len(tr_classes)
        for tr in tr_classes:
            td_classes = tr.find_all('td')
            row = []
            for td_class in td_classes:
                row.append(td_class.text.strip())
            assert(len(row)==len(COLUMNS))
            entry_id = row[0]
            if entry_id in data:
                #print "Already found:", entry_id
                continue
            else:
                #print entry_id
                data[entry_id] = row
        time.sleep(1)

    df = pd.DataFrame.from_records(data.values(), columns=COLUMNS)
    del df['delete_node']

    df['unique_id'] = df['unique_id'].astype(int)
    df = df[df['unique_id']!=0]
    df = df.sort_values(['pub_year', 'unique_id'])
    df = df.reset_index(drop=True)
    df.to_csv('usp.csv', encoding='utf-8', index=False)

    pairs = df.groupby(['ingredient', 'adulterant']).size().sort_values()[::-1]
    print 'Number of entries  :', len(df)
    print 'Unique entries     :', len(pairs)
    print 'Unique ingredients :', len(df['ingredient'].unique())
    print 'Unique adulterants :', len(df['adulterant'].unique())
    pairs.to_csv('usp_pairs.csv', encoding='utf-8', header=True)

def read_usp_data():
    return pd.read_csv('usp.csv')

