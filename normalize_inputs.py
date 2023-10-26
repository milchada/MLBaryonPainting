def renorm(arr, arrmin, arrmax, eps=0.1, log=True):
    if log:
        ret = np.log10(arr) #log10(Lx)
        ret[ret == -np.inf] = np.nan 
    else:
        ret = arr
        arrmin = 10**arrmin 
        arrmax = 10**arrmax
    ret -= arrmin #min(log10(Lx))
    ret += eps
    ret /= (arrmax-arrmin+eps) #max(log10(Lx))
    return ret #to reconstruct arr, need np.nanmin(arr), np.nanmax(arr)
