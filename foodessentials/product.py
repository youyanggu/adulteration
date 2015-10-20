class Product(object):
    def __init__(self, p):
        self.p = p

    @property
    def ingredients(self):
        return self.p['ingredients']

    @property
    def upc(self):
        return self.p['upc']

    @property
    def name(self):
        return self.p['product_name']

    @property
    def category(self):
        return self.p['food_category']

    def __str__(self):
        return self.name