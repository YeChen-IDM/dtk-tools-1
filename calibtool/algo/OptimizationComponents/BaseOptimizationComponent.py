import numpy as np

class BaseOptimizationComponent(object):

    def MapInToOut(self, x):
        print 'User to implement MapInToOut'
        return x

    def MapOutToIn(self, x):
        print 'User to implement MapOutToIn'
        return x

    def _convert_value(self, value):
        if isinstance(value, list):
            return np.asarray(value)
        elif isinstance(value, np.ndarray):
            return value
        elif isinstance(value, int) or isinstance(value, float):
            return np.asarray( [value] )

        print 'Teach me how to convert a value of type', type(value) # TODO: logger
        exit()  # TODO: error out

    def __init__(self, **kwargs):
        self.config = kwargs;

        x_initial = self._convert_value( kwargs['x_initial'] )

        has_x_current = 'x_current' in kwargs
        if has_x_current:
            x_current = self._convert_value( kwargs['x_current'] )

        low = self._convert_value( kwargs['lower_bound'] )
        high = self._convert_value( kwargs['upper_bound'] )
        self.name = kwargs['name']

        self.lower_bound = low
        self.upper_bound = high

        self.is_dynamic = True
        if 'is_dynamic' in kwargs and not kwargs['is_dynamic']:
            print 'Creating static component: %s' % self.name
            self.is_dynamic = False

        self.json_paths = kwargs['json_paths'];
        self.labels = kwargs['labels']

        self.dim_in = x_initial.shape[0]
        self.dim_out = len(self.json_paths)

        # Finally, set the current value of x
        if has_x_current:
            self.set_x_current( x_current )
        else:
            self.set_x_current( x_initial )

    def set_x_current(self, x):
        x = self._convert_value(x)

        # Keep current postmap in sync with current
        self.x_current_postmap = self.MapInToOut( x )

        # Map through to get range clamping
        self.x_current = self.MapOutToIn( self.x_current_postmap )

    #def serialize(self):
    #    return pickle.dumps(self))
