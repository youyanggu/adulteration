def standardize_ingredient(i):
    for sym in [':', '.', '*', '(', ')']:
        i = i.strip()
        i = i.strip(sym)
    i = ' '.join(i.split()) # remove multiple spaces
    return i

def parse_ingredients(s, is_subingredient=False):
    s = s.strip()
    s = ' '.join(s.split()) # remove multiple spaces
    s = s.lower()
    if len(s) <= 1:
        return []
    for sym in ['[', '{']:
        s = s.replace(sym, '(')
    for sym in [']', '}']:
        s = s.replace(sym, ')')
    if '. contains' in s:
        s = s.replace('. contains', ', contains')
    if '(from concentrate)' in s or '(concentrate)' in s:
        s = s.replace('(from concentrate)', 'from concentrate')
        s = s.replace('(concentrate)', 'from concentrate')
    if 'cocoa (processed with alkali)' in s:
        s = s.replace('cocoa (processed with alkali)', 'cocoa processed with alkali')
    if is_subingredient:
        for term in ['to', 'for', 'as an', 'as a', 
                     'added to', 'added for', 'maintains']:
             if s.startswith('{} '.format(term)):
                  return []
        if s in []:
             return ['milk']
        if s in ['for color', 'flavor enhancers', 'leavening agents', 
                 'leavening', 'dried', 'live', 'natural sweetener',
                 'preserve freshness', 'an artificial flavor',
                 'if colored', 'artificial flavor', 'natural flavors',
                 'flavor enhancer', 'natural', 'artificial flavor',
                 'color', 'preservative', 'artificial flavors',
                 'artificial color', 'preservatives', 'natural flavoring',
                 'flavor', 'color added', 'as a preservative', 
                 'natural artificial flavors', 'added color',
                 'added to prevent caking', 'to prevent caking',
                 'as a preservative', 'artificial colors', 'colors',
                 'artificial', 'artificial flavoring',
                 'a source of calcium', 'a milk derivative',
                 'processed with alkali', 'a preservative',
                 'an emulsifier', 'a natural mold inhibitor',
                 'preserves freshness', 'as preservatives',
                 'anticaking agent', 'sweetener', 'used to protect quality',
                 'controls acidity', 'prevents caking', 'anti-caking agent',
                 'provides tartness'
                 'and', 'or']:
             return []
        if ',' not in s:
             if s[-1] == '%':
                  return []
             if s.startswith('from '):
                  s.replace('from ', '')
    for phrase in ['and', 'and/or', 'or']:
        if ', {} '.format(phrase) in s:
             s = s.replace(', {} '.format(phrase), ', ')
        if phrase == 'and/or':
             s = s.replace(' {} '.format(phrase), ', ')
    for phrase in [
                   'contains 2%, less of each of the following',
                   'contains 2%, less of the following',
                   'contains 2%, less of',
                   'contains 2% or less of the following', 
                   'contains 2% or less of',
                   'contains 2% or less',
                   'contains less than 2% of the following',
                   'contains less than 2% of',
                   'contains less than 2%',
                   'less than 2% of the following',
                   'less than 2% of',
                   'less than 2%',
                   'contains 2% of less of the following',
                   'contains 2% of less of',
                   'contains 2% of less',
                   'contains 2% less of the following',
                   'contains 2% less of',
                   'contains 2% less',
                   'contains <2% of',
                   'contains <2%',
                   '2% or less of',
                   '2% less of',
                   'contains 2 % or less of',
                   'contains 2 percent less of',
                   'contains 2 percent or less of',
                   'contains two percent or less of',
                   'contains two percent less of',
                   'contains less than 1% of',
                   'contains 1% or less of',
                   'contains 1% less of',
                   'contains 0. 5% less of',
                   'contains one or more of the following',
                   'contains one or more of',
                   'contains one more of the following',
                   'contains one more of',
                   'more of the following',
                   'the following',
                   'other ingredients',
                   'ingredients',
                   'contains',
                   'including',
                   ]:
        if s.startswith(phrase) or ', {}'.format(phrase) in s:
             s = s.replace(phrase, '')
    ingredients_split = re.findall(r'([^,\(]*)\((.*?)\)', s)
    middle_ingredients = [i[0] for i in ingredients_split]
    subingredients = [parse_ingredients(i[1], True) for i in ingredients_split]
    subingredients = list(itertools.chain.from_iterable(subingredients))
    main_ingredients = re.sub(r'\([^)]*\)', '', s)
    main_ingredients_split = main_ingredients.split(', ')
    main_minus_middle_ingredients = [i for i in main_ingredients_split if i not in middle_ingredients]
    all_ingredients = main_ingredients_split + subingredients
    return all_ingredients

