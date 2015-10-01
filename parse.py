import numpy as np

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
        return product[0]
    if len(product) == 2:
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
               'high content of', 'abnormal smell of', 'unauthorised']
    chem = chemical
    for p in phrases:
        chem = chem.replace(p, '')
    chem = ' '.join(chem.split())
    if chem.startswith('and '):
        chem = chem[4:]
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