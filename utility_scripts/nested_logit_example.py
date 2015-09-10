def getMevForMultiLevelNested(V, av, tree, param, rootcode=0):
    '''Source: djarpin/BIObox on GitHub'''
    availability = av.copy()
    y = {}
    for i, v in V.items():
        y[i] = exp(v)

    Gi = {}

    def ifavail(i, val):
        return Elem({0: 0.0, 1: val}, availability[i] != 0)

    param[rootcode] = 1.0
    tree[rootcode] = rootcode
    for parent in param.keys():
        availability[parent] = 0
        y[parent] = 0.0
    # calculate y on up
    for child, parent in tree.items():
        if child in param:
            y[child] = y[child] ** (1.0/param[child])
        y[parent] += ifavail(child, y[child] ** param[parent])
        availability[parent] = Elem({0: availability[child], 1: 1},
                                    availability[parent] != 0)
    # calculate Gi
    for i in V.keys():
        n = tree[i]
        Gi[i] = y[i]**(param[n]-1.0)
        nn = tree[n]
        while nn != n:
            Gi[i] *= y[n]**(param[nn] - param[n])
            n = nn
            nn = tree[n]
        Gi[i] = Elem({0: (0), 1: Gi[i]}, availability[i] != 0)
    return Gi
