import re

def standardize_ingredient(s):
    s = s.strip(':.*-(), ')
    s = ' '.join(s.split()) # remove multiple spaces
    if len(s)<=1:
        return ''
    """
    if s in ['flavor enhancers', 'leavening agents', 
             'leavening', 'dried', 'natural sweetener',
             'an artificial flavor',
             'if colored', 'artificial flavor', 'natural flavors',
             'flavor enhancer', 'natural', 'artificial flavor',
             'color', 'preservative', 'artificial flavors',
             'artificial color', 'preservatives', 'natural flavoring',
             'flavor', 'color added', 'a natural preservative',
             'natural artificial flavors',
             'artificial colors', 'colors',
             'artificial flavoring',
             'a milk derivative',
             'a milk ingredient',
             'a preservative', 'a natural thickener',
             'an emulsifier', 'a natural mold inhibitor',
             'anticaking agent', 'sweetener',
             'anti-caking agent',
             ]:
         return ''
    """
    if s in ['made', 'and', 'or']:
         return ''
    for term in ['preserves', 'prevents', 'prevent',
                 'provides', 'preserve', 'controls', 
                 'maintains']:
         if s.startswith('{} '.format(term)):
              return ''
    if s[-1] == '%':
         s = re.sub(r'[0-9. ]+\%', '', s)
    return s

def parse_ingredients(s):
    s = s.strip()
    s = ' '.join(s.split()) # remove multiple spaces
    s = s.lower()
    if len(s) <= 1:
        return [], []
    s = s.replace('[', '(')
    s = s.replace('}', '(')
    s = s.replace(']', ')')
    s = s.replace('}', ')')
    s = s.replace('; ', ', ')
    s = s.replace('0. 1%', '0.1%')
    s = s.replace('0. 5%', '0.5%')
    s = s.replace('org.', 'organic')
    s = s.replace('a source of', '')
    for term in ['salt', 'water', 'enzymes', 'sugar', 'preservative', 
                 'color', 'carrageenan', 'spices',
                 'milk', 'dextrose', 'spice', 'yeast', 'soy', 'riboflavin',]:
        s = s.replace(term+'.', term+',')
    if '(from concentrate)' in s or '(concentrate)' in s:
        s = s.replace('(from concentrate)', 'from concentrate')
        s = s.replace('(concentrate)', 'from concentrate')
    for term in ['added to', 'added for', 'used', 'to', 'as', 'for']:
        if '{} '.format(term) in s:
            s = re.sub(r' {} [A-Za-z ]+'.format(term), '', s)
            s = re.sub(r'\({} [A-Za-z ]+\)'.format(term), '', s)
    if '(processed with alkali)' in s:
        s = s.replace('(processed with alkali)', 'processed with alkali')
    for phrase in ['and', 'and/or', 'or']:
        if ', {} '.format(phrase) in s:
             s = s.replace(', {} '.format(phrase), ', ')
        if phrase == 'and/or':
             s = s.replace(' {} '.format(phrase), ', ')
    for phrase in [
                   'contains 2% or less of',
                   'contains 2% or less',
                   'contains less than 2% of',
                   'contains less than 2%',
                   'contains 2% of less of',
                   'contains 2% of less',
                   'contains 2% less of',
                   'contains 2% less',
                   'contains <2% of',
                   'contains 2 % or less of',
                   'contains 2 percent or less of',
                   'contains two percent or less of',
                   'contains less than 1% of',
                   'contains 1% or less of',
                   'contains one or more of',
                   'contains one more of',
                   'contains',
                   'contain:',
                   '2% or less of',
                   '2% less of',
                   'less than 2% of',
                   'less than 2%',
                   'more of the following',
                   'each of the following',
                   'the following',
                   'other ingredients',
                   'ingredients',
                   'including',
                   'filling:',
                   'of:'
                   ]:
        if phrase not in s:
             continue
        if s.startswith(phrase):
             s = s.replace(phrase, '')
        else:
             s = re.sub(r'[.,\(][ ]*{}'.format(phrase), ', ', s)
    middle_ingredients = re.findall(r'([^,\(]*)\(', s)
    s_flat = s.replace('(', ', ').replace(')', '')
    all_ingredients = s_flat.split(', ')
    return all_ingredients, middle_ingredients

