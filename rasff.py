import csv
import numpy as np
import pandas as pd


###################
# Load CSV file
###################

fname = '../rasff_ic.csv'

f = open(fname, 'rU')
reader = csv.reader(f)

header = next(reader)
header = ['#', 'classification', 'date_case', 'date_last_change', 'reference', 
          'country', 'type', 'category', 'subject']

count = 1
first = True
rows = []
curRow = []
for line in reader:
    if first:
        curRow.extend(line)
    else:
        curRow.extend(line[1:2])
    if not first:
        rows.append(curRow)
        curRow = []
    first = not first

df = pd.DataFrame.from_records(np.array(rows), exclude='#', columns=header)
food = df[df['type'].str.lower()=='food']
food['type'] = 'food'
food['subject'] = food['subject'].str.replace('\xe5\xb5', 'u')
food.reset_index(drop=True, inplace=True)
food.to_csv('../rasff_ic_clean.csv')

##################
# Analysis
##################

df = pd.DataFrame.from_csv('../rasff_ic_clean.csv')
subject = df.subject.values
chemicals = [i.split(' (')[0] for i in subject]

d = {}
for i,c in enumerate(chemicals):
    key = (c, df.category.ix[i])
    if key not in d:
        d[key] = 1
    else:
        d[key] += 1

d_filtered = {k : d[k] for k in d if d[k]>1}

