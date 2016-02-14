import json
import time

import cirpy
cirpy.API_BASE = 'http://cactus.nci.nih.gov/chemical/structure'

def find_iupac_names():
    df_ = read_df()
    chems = df_.groupby('chemical_').size().sort_values()[::-1].index.values
    iupac_names = []
    num_found = 0
    for i, chem in enumerate(chems):
        if not chem:
            continue
        print chem
        iupac = cirpy.resolve(chem, 'iupac_name')
        if iupac:
            print "Found:", iupac
            num_found += 1
        else:
            print "Not found."
        iupac_names.append(iupac)
        print '{} / {}'.format(num_found, i+1)
        print ''
        time.sleep(1)
    return iupac_names

def read_chemnet():
    chem_names = []
    with open('chemicals.json', 'rb') as f:
        count = 0
        for line in f:
            if count % 100000 == 0:
                print count, len(chem_names)
            chem = json.loads(line)
            if 'smiles' not in chem or not chem['smiles']:
                continue
            chem_names.append(chem['smiles'])
            #for name in chem['name']:
            #    chem_names.append(name)
            count += 1
    return set(chem_names)

df = pd.read_csv('../misc/all_adulterants_processed.csv')
smiles = df['auto SMILES'].unique()
found_adulterants = set()
founds = []
for i, row in df.iterrows():
    if row['adulterant'] in found_adulterants:
        founds.append(False)
        continue
    if row['auto SMILES'] in chem_names:
        print '{:<25} {:<10} {}'.format(row['adulterant'], row['source'], row['auto SMILES'])
        founds.append(True)
    else:
        founds.append(False)
    found_adulterants.add(row['adulterant'])