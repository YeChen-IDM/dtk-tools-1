import json, pickle
from SimpleComponent import SimpleComponent

config = {
    'name': 'Dan',
    'x_initial': 1,
    'x_current': 3,
    'lower_bound': 0,
    'upper_bound': 10,
    'is_dynamic': True,
    'json_paths': [
        'JP'
    ],
    'labels': [ 'JPL' ]
}

comp = SimpleComponent( **config )
comp.set_x_current(1e3)
s = pickle.dumps(comp)

comp2 = pickle.loads(s)
print comp2.x_current
