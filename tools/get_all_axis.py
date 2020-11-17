def get_all(arr):
    # get all possible arrays given by taking tha negation of some coordinates of arr
    dim = arr.shape[0]
    vec = []
    for i in range(dim):
        if i == 0:
            if arr[i] != 0:
                k = int(arr[0])
                vec.append(np.array([k]))
                vec.append(np.array([-k]))
            else:
                vec.append(np.array([0]))

        else:
            if int(arr[i]) != 0:
                for j in range(len(vec)):
                    v = vec[j]
                    l = np.copy(v)
                    k = int(arr[i])
                    k = np.array([k])
                    vec[j] = np.concatenate((v, k))
                    l = np.concatenate((l, -k))
                    vec.append(l)
            else:
                for i in range(len(vec)):
                    v = vec[i]
                    vec[i] = np.concatenate((v, np.array([0])))
    return vec

def get_all_axis(dim, thres):
    ## dim: dimension of each output vector
    ## thres:
    ## function returns all vectors of dimension dim such that each
    ## coordinate of the vector takes integer value and
    vecs, vecs1 = [], []
    ind = 0
    for i in range(thres * dim):
        if i == 0:
            vecs.append(np.zeros(dim))
        else:
            k = len(vecs)
            c = set([])
            for h in range(ind, k):
                l = vecs[h]
                for j in range(dim):
                    if l[j] < thres:
                        f = np.copy(l)
                        f[j] = f[j] + 1
                        f = f.tostring()
                        c.add(f)
            ind = k
            for element in c:
                element = np.fromstring(element)
                vecs.append(element)
    for v in vecs:
        vecs1 = vecs1 + get_all(v)
    return vecs1
