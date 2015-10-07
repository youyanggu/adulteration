import numpy as np

def clean_products(df):
    alt_names = pd.DataFrame.from_csv('{}/alternate_names.csv'.format(rasff_dir), header=None).ix[:,1].to_dict()
    df['product'] = df['product'].str.lower()
    products = np.unique(df['product'].values)
    replace = []
    d = {i:True for i in products}
    df['product'] = [alt_names[p] if p in alt_names else p for p in df['product']]
    for p in products:
        if p+'s' in d:
            replace.append(p)
    for r in replace:
        df['product'] = df['product'].str.replace(r+'s', r)
    return df

def clean_chemicals(df):
    return df

def parse_origin(origin):
    origin = origin[1:]
    if origin == []:
        return ''
    if len(origin) == 1:
        if 'containing unauthorised' in origin[0]:
            return origin[0].split(' containing ')[0]
        return origin[0]
    if len(origin) > 1:
        if origin[0].islower() and origin[0][:3] != 'the':
            return ' from '.join(origin[1:])
        else:
            return ' from '.join(origin)

def parse_product(product):
    if len(product) == 0:
        return ''
    if len(product) == 1:
        if 'placing on the market of' in product[0] or \
            'carbon monoxide treatment' in product[0]:
            return product[0].split(' of ')[-1]
        if 'adverse reaction caused by' in product[0]:
            return product[0].split(' by ')[-1]
        return product[0]
    if len(product) == 2:
        if 'insufficient labelling of food supplement' in product[1]:
            return 'food supplement'
        return product[1]
    if len(product) > 2:
        s = ' in '.join(product[1:])
        return s.split(' manufactured in ')[0]

def parse_amount(amount):
    if len(amount) <= 1:
        return ''
    if len(amount) >= 2:
        # if more than 2 chemicals, just take first one for now
        return amount[1].split(')')[0]

def parse_chemical(chemical):
    phrases = ['unauthorised use of', 'too high content of', 'undeclared',
               'suffocation risk as a result of the consumption of',
               'unauthorised substances', 'unauthorised substance',
               'high content of', 'abnormal smell of', 'unauthorised',
               'addition of', 'suspicion of', 'presence of']
    chem = chemical
    for sub in ['carbon monoxide treatment', 'adverse reaction', 'placing on the market']:
        if sub in chem:
            return ''
    chem = chem.replace('colours', 'colour')
    for p in phrases:
        chem = chem.replace(p, '')
    chem = ' '.join(chem.split())
    if chem.startswith('and '):
        chem = chem[4:]
    if ' and ' in chem:
        return chem.split(' and ')[0]
    return chem

def parse_subject(sub):
    chemical, amount, product, origin = [], [], [], []
    sub = [' '.join(i.split()) for i in sub]

    # origin
    split = [i.split(' from ') for i in sub]
    for v in split:
        origin.append(parse_origin(v))

    # product
    split = [i[0] for i in split]
    split = [i.split(' in ') for i in split]
    for v in split:
        product.append(parse_product(v))

    # amount
    split = [i[0] for i in split] 
    split = [i.split(' (') for i in split]
    for v in split:
        amount.append(parse_amount(v))

    # chemical
    split = [i[0] for i in split]
    for v in split:
        chemical.append(parse_chemical(v))

    assert(len(chemical)==len(amount)==len(product)==len(origin)==len(sub))
    return chemical, amount, product, origin

