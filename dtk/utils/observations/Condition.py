class Condition:
    def __init__(self, stratifier, op, value):
        self.stratifier = stratifier
        self.op = op
        self.value = value


    def apply(self, df):
        return self.op(df[self.stratifier], self.value)


    def __str__(self):
        return '%s %s %s' % (self.stratifier, self.op, self.value)
