import numpy as np

from BaseOptimizationComponent import BaseOptimizationComponent

class SimpleComponent(BaseOptimizationComponent):

    def __init__(self, **kwargs):
        super(SimpleComponent, self).__init__(**kwargs)

    def Constrain(self, x_in):
        assert( x_in.ndim == self.dim_in);

        x_out = np.minimum(self.upper_bound, np.maximum(self.lower_bound, x_in));

        x_out_bounds = [(xi,xo) for (xi,xo) in zip(x_in, x_out) if xi < self.lower_bound or xi > self.upper_bound]
        for x in x_out_bounds:
            print 'Component %s with x_in=%f was not in range [%f,%f] --> x_out=%f\n' % ( self.name, x[0], self.lower_bound, self.upper_bound, x[1] )

        assert( x_out.ndim == self.dim_out)
        return x_out

    def MapOutToIn(self, x_out):
        x_in = x_out
        return x_in

