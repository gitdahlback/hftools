#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
# -*- coding: utf-8 -*-
import numpy as np
from hftools.dataset import hfarray, DimMatrix_i, DimMatrix_j,\
    DimUMatrix_i, DimUMatrix_j


class Sensitivities(object):
    def __init__(self, **k):
        self._sens = k

    def add_sens(self, other):
        out = Sensitivities(**self._sens)
        for k, v in other._sens.items():
            out._sens[k] = out._sens.get(k, 0) + v
        return out

    def multiply_sens(self, value):
        out = Sensitivities()
        for k, v in self._sens.items():
            out._sens[k] = v * value
        return out

    def __neg__(self):
        out = Sensitivities()
        for k, v in self._sens.items():
            out._sens[k] = -v
        return out

    def __repr__(self):
        return self._sens.__repr__()

    @property
    def slice(self):
        return Slicer(self)

    def items(self):
        return self._sens.items()


class Slicer(object):
    def __init__(self, sens):
        self.sens = sens

    def __getitem__(self, items):
        out = {}
        for k, v in self.sens.items():
            out[k] = v[items]
        return out




class UncertainValue(object):
    def __init__(self, name, value, uncertainties, sens=None):
        self.name = name
        self.value = value
        self.uncertainties = uncertainties
        if name is None:
            self.sens = Sensitivities() if sens is None else sens
        else:
            self.sens = Sensitivities(**{name: hfarray(1.)})

    def __neg__(self):
        out = UncertainValue(None, -self.value,
                             self.uncertainties, sens=-self.sens)
        return out

    def __add__(self, other):
        if isinstance(other, UncertainValue):
            value = self.value + other.value
            sens = self.sens.add_sens(other.sens)
            unc = other.uncertainties
            unc.update(self.uncertainties)
            out = UncertainValue(None, value, unc, sens=sens)
        else:
            out = UncertainValue(None, self.value + other,
                                 self.uncertainties, self.sens)
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, UncertainValue):
            value = self.value * other.value
            sens1 = self.sens.multiply_sens(other.value)
            sens2 = other.sens.multiply_sens(self.value)
            sens = sens1.add_sens(sens2)
            unc = other.uncertainties
            unc.update(self.uncertainties)
            out = UncertainValue(None, value, unc, sens=sens)
        else:
            sens = self.sens.multiply_sens(other)
            out = UncertainValue(None, self.value * other,
                                 self.uncertainties, sens)
        return out

    def __rmul__(self, other):
        sens = self.sens.multiply_sens(other)
        out = UncertainValue(None, self.value * other,
                             self.uncertainties, sens)
        return out

    def __div__(self, other):
        if isinstance(other, UncertainValue):
            value = self.value / other.value
            sens1 = self.sens.multiply_sens(1 / other.value)
            sens2 = other.sens.multiply_sens(-self.value / other.value ** 2)
            sens = sens1.add_sens(sens2)
            unc = other.uncertainties
            unc.update(self.uncertainties)
            out = UncertainValue(None, value, unc, sens=sens)
        else:
            sens = self.sens.multiply_sens(1 / other)
            out = UncertainValue(None, self.value / other,
                                 self.uncertainties, sens)
        return out

    def __rdiv__(self, other):
        sens = self.sens.multiply_sens(-other / self.value ** 2)
        out = UncertainValue(None, other / self.value,
                             self.uncertainties, sens)

        return out

    def __pow__(self, other):
        if isinstance(other, UncertainValue):
            value = self.value ** other.value
            c1 = self.value ** (-1 + other.value) * other.value
            c2 = self.value ** other.value * np.log(self.value)
            sens1 = self.sens.multiply_sens(c1)
            sens2 = other.sens.multiply_sens(c2)
            sens = sens1.add_sens(sens2)
            unc = other.uncertainties
            unc.update(self.uncertainties)
            out = UncertainValue(None, value, unc, sens=sens)
        else:
            c1 = self.value ** (-1 + other) * other
            sens = self.sens.multiply_sens(c1)
            out = UncertainValue(None, other ** self.value,
                                 self.uncertainties, sens)
        return out

    def __rpow__(self, other):
        c1 = other ** self.value * np.log(other)
        sens = self.sens.multiply_sens(c1)
        out = UncertainValue(None, other ** self.value,
                             self.uncertainties, sens)
        return out


def cast_to_uncertainty_matrix(x):
    X = x

    if DimUMatrix_i not in X.dims and DimMatrix_i in X.dims:
        old = X.dims.get_matching_dim(DimMatrix_i)
        X = X.replace_dim(old, DimUMatrix_i(old, "I"))
    if DimUMatrix_j not in X.dims and DimMatrix_j in X.dims:
        old = X.dims.get_matching_dim(DimMatrix_j)
        X = X.replace_dim(old, DimUMatrix_j(old, "J"))
    return X


def uv(name, value, uncertainty):
    U = cast_to_uncertainty_matrix(uncertainty)
    return UncertainValue(name, value, {name: U})

