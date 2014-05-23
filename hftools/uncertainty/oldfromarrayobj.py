import numpy as np
import numpy.lib.stride_tricks as np_stride_tricks


def info_has_complex(info):
    deprecate("info_has_complex is deprecated")
    return dims_has_complex(info)


def dims_has_complex(dims):
    for dim in reversed(dims):
        dims = (ComplexDerivAxis, ComplexIndepAxis, ComplexDiagAxis)
        if isinstance(dim, dims):
            return True
    return False


def as_strided(x, shape=None, strides=None, offset=0):
    u"""Low-level routine to reshape an array by specifying shape, strides,
        and offset.

        Parameters
        ----------
          *shape*   is a list that specifies how many elements there are in
                    each dimension

          *strides* is a list that specifies how many bytes to move when
                    increasing the index of the dimension by one.

          *offset*  specifies the offset from the starting point of the
                    buffer to the first element, i.e. when all indices are
                    zero.

        Use only if you truly understand how arrays are stored in memory.
        See numpy documentation for more information.
    """
    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)
    if offset:
        addr, boolflag = interface['data']
        addr += offset
        interface['data'] = addr, boolflag
    return np.asarray(np_stride_tricks.DummyArray(interface, base=x))


def diagonal_view(x, diagonal):
    u"""Give view of array *x* where we index the diagonal of the dimensions
    specified by the *Dim* objects in the *diagonal* tuple.

    The diagonal index will have the same position as the first index in
    *diagonal*
       >>> fi = DiagAxis("Frekvens[Hz]", np.linspace(0, 400e6, 3))
       >>> cf = hfarray(np.zeros((3, 3)), (fi, fi.deriv_axis))
       >>> diag = diagonal_view(cf, cf.dims)
       >>> diag[...] = np.arange(0, 3) + 11
       >>> cf
       hfarray([[ 11.,   0.,   0.],
                [  0.,  12.,   0.],
                [  0.,   0.,  13.]])
    """
    diagstride = 0
    diagshapes = []
    newstrides = list(x.strides[:])
    newshapes = list(x.shape[:])

    indices = [x.dims.index(ax) for ax in diagonal]
    first = indices[0]

     #reversed so we can delete idx without messing up
     #the order of the remaining elements
    indices.sort(reverse=True)
    for idx in indices:
        diagstride += newstrides[idx]
        diagshapes.append(newshapes[idx])

        del newstrides[idx]
        del newshapes[idx]
    newshapes.insert(first, min(diagshapes))
    newstrides.insert(first, diagstride)
    return as_strided(x, newshapes, newstrides)


def _expand_diagonal(x, diagaxis):
    u"""Helper function that expands DiagAxis dimensions to a
    (IndepAxis, DerivAxis) pair
    """
    if not isinstance(diagaxis, (_DiagAxis, )):
        raise Exception("diagaxis must be a _DiagAxis")

    if diagaxis not in x.dims:
        raise Exception("%s does not contain a %r axis" % (x, diagaxis))

    idx = x.dims.index(diagaxis)
    diagaxis = x.dims[idx]
    newinfo = list(x.dims[:])
    newshape = list(x.shape[:])
    newinfo[idx:idx + 1] = [diagaxis.indep_axis, diagaxis.deriv_axis]
    newshape[idx:idx + 1] = [diagaxis.indep_axis.data.shape[0], ] * 2
    out = x.__class__(np.zeros(newshape, dtype=x.dtype), tuple(newinfo))
    diagonal_view(out, newinfo[idx:idx + 2])[...] = x
    return out


def expand_diagonals(x, diags=None):
    u"""Expanderar en DiagAxis dimensioner i *diags* till IndepAxis, DerivAxis
    dimensioner. Om *diags*=None sa expandera alla DiagAxis dimensioner
    """
    out = x
    if diags is None:
        diags = [ax for ax in x.dims if isinstance(ax, _DiagAxis)]

    for ax in diags:
        out = _expand_diagonal(out, ax)
    return out



def make_fullcomplex_array(a):
    cls = a.__class__
    if isfullcomplex(a):
        out = a.view()
    else:
        out = np.zeros(a.shape + (2, 2), dtype=np.float64)
        try:
            dims = a.dims + CPLX
        except AttributeError:
            dims = CPLX
            cls = hfarray
        A = a.view(type=np.ndarray, dtype=a.dtype)
        if np.iscomplexobj(a):
            out[..., 0, 0] = out[..., 1, 1] = A.real
            out[..., 0, 1] = -A.imag
            out[..., 1, 0] = A.imag
        else:
            out[..., 0, 0] = A.real
        out = cls(out, dims=dims, copy=False)
    return out


def isfullcomplex(x):
    try:
        return dims_has_complex(x.dims)
    except AttributeError:
        return False
