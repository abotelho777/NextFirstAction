import numpy as np
from scipy import misc as msc
import os
import math

from scipy.stats import bernoulli,norm


def bern(p,shape):
    return bernoulli.rvs(p,size=shape)


def norm(shape):
    return norm.rvs(size=shape)


def normalize(v, method='max'):
    m = np.nanmean(v)
    s = np.nanstd(v)
    mx = np.nanmax(v)
    if mx == 0:
        mx = 0.000001
    me = np.nanmedian(v)
    mi = np.nanmin(v)
    nv = []
    for i in range(0,len(v)):
        if method == 'max':
            nv.append(v[i]/mx)
        elif method == 'zscore':
            if m == 0 and s == 0:
                nv.append(0)
            else:
                nv.append((v[i]-m)/s)
        elif method == 'median':
            nv.append(v[i]/me)
        elif method == 'uniform':
            nv.append((v[i]-mi)/(mx-mi))
        else:
            print("ERROR - UNKNOWN METHOD")

    return nv


def Aprime(actual, predicted):
    assert len(actual) == len(predicted)

    score = [[],[]]

    for i in range(0,len(actual)):
        score[int(actual[i])].append(predicted[i])

    sum = 0.0
    for p in score[1]:
        for n in score[0]:
            if p > n:
                sum += 1
            elif p == n:
                sum += .5
            else:
                sum += 0

    return sum/(float(len(score[0]))*len(score[1]))


def loadIMG(filename,grayscale=0):
    try:
        img = msc.imread(filename,grayscale)
    except ValueError:
        print("ERROR loading file:",filename)
        return []
    return img


def getfilenames(extension=".*",directory=os.path.dirname(os.path.realpath(__file__))):
    names = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            names.append(directory + "/" + file)
    return names


def loadCSV(filename, max_rows=None):
    csvarr = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            # split out each comma-separated value
            name = line.strip().split(',')
            for j in range(0,len(name)):
                # try converting to a number, if not, leave it
                try:
                    name[j] = float(name[j])
                except ValueError:
                    # do nothing constructive
                    name[j] = name[j]
            csvarr.append(name)
            if max_rows is not None:
                if len(csvarr) >= max_rows:
                    break
    return csvarr


def writetoCSV(ar,filename,headers=[]):
    # ar = np.array(transpose(ar))
    np_ar = np.array(ar)
    assert len(np_ar.shape) <= 2

    with open(filename + '.csv', 'w') as f:
        if len(headers)!=0:
            for i in range(0,len(headers)-1):
                f.write(str(headers[i]) + ',')
            f.write(str(headers[len(headers)-1])+'\n')
        for i in range(0,len(ar)):
            if len(np_ar.shape) == 2:
                for j in range(0,len(ar[i])-1):
                    val = str(ar[i][j])
                    if type(ar[i][j]) is str:
                        val = '=\"' + val + '\"'
                    f.write(val + ',')
                f.write(str(ar[i][len(ar[i])-1]) + '\n')
            else:
                val = str(ar[i])
                if type(ar[i]) is str:
                    val = '=\"' + val + '\"'
                f.write(val + '\n')
    f.close()


def loadCSVwithHeaders(filename, max_rows=None):
    if max_rows is not None:
        max_rows += 1
    data = loadCSV(filename,max_rows)
    headers = np.array(data[0])
    data = np.array(data)
    data = np.delete(data, 0, 0)
    return data,headers


def readHeadersCSV(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            # split out each comma-separated value
            return line.strip().split(',')
    return []


def factorial(x):
    print x
    val = x
    for i in range(1,x-1):
        print x-i
        val *= x-i
    return val


def getPermutations(ar, as_str=False):

    def perm(ar, ind):
        if ind == 1:
            return [[i] for i in ar]
        t = perm(ar,ind-1)
        p = list(t)
        tprime = []
        for i in range(0,len(t)):
            if len(t[i]) == ind-1:
                    tprime.append(t[i])
        for i in range(0,len(tprime)):
            head = tprime[i]
            for j in range(ar.index(max(head))+1,len(ar)):
                c = list(head)
                c.append(ar[j])
                p.append(c)
        return p

    ret = perm(ar,len(ar))

    if not as_str:
        return ret
    else:
        for i in range(0,len(ret)):
            s = ''
            for j in range(0,len(ret[i])):
                s += str(ret[i][j])
            ret[i] = s
        return ret


def convert_to_floats(ar):
    data = []
    for i in range(0, len(ar)):
        row = []
        for j in range(0, len(ar[i])):
            k = 0
            # try converting to a number, if not, leave it
            try:
                k = float(ar[i][j])
            except ValueError:
                # do nothing constructive
                k = ar[i][j]
            row.append(k)
        data.append(row)

    return data


def getColumn(ar,col,startRow = 0):
    c = []

    # get number of rows and columns
    numR = len(ar)
    # return if index out of bounds
    if startRow >= numR:
        print("INDEX OUT OF BOUNDS")
        return c
    numC = len(ar[startRow])
    # return if index out of bounds
    if col >= numC:
        print("INDEX OUT OF BOUNDS")
        return c



    # create an array from the column values
    for i in range(startRow, numR-1):
        c.append(ar[i][col])



    return c


def unique(ar,ignore=list([])):
    ulist = []

    assert type(ignore) is list

    for i in range(0,len(ar)):
        if ar[i] not in ignore and ar[i] not in ulist:
            ulist.append(ar[i])

    return ulist


def fold(x,y,folds=2):
    assert folds > 0
    assert len(x) == len(y)
    f = []
    f_l = []

    for i in range(0,folds):
        f.append([])
        f_l.append([])

    for i in range(0,len(x)):
        rindex = np.random.randint(0, folds)
        f[rindex].append(x[i])
        f_l[rindex].append(y[i])

    return np.array(f),np.array(f_l)


def split_training_test(x,y,training_size=0.8):
    assert len(x) == len(y)

    x,y = shuffle(x,y)

    X_Train = []
    X_Test = []
    Y_Train = []
    Y_Test = []

    for i in range(0,int(len(x)*training_size)):
        X_Train.append(x[i])
        Y_Train.append(y[i])
    for i in range(len(X_Train)+1,len(x)):
        X_Test.append(x[i])
        Y_Test.append(y[i])

    return np.array(X_Train),np.array(X_Test),np.array(Y_Train),np.array(Y_Test)


def sample(x,y,p=0.8,n=None):
    assert p > 0 and p <= 1
    assert len(x) == len(y)

    X = []
    X_label = []

    x,y = shuffle(x,y)

    if n is None:
        for i in range(0,len(x)):
            from random import randint
            if randint(0,100) < int(100*p)-1:
                X.append(x[i])
                X_label.append(y[i])
    else:
        while len(X) < n:
            for i in range(0, len(x)):
                from random import randint
                if randint(0, 100) < int(100 * p) - 1:
                    X.append(x[i])
                    X_label.append(y[i])
                if len(X) == n:
                    break

    return np.array(X),np.array(X_label)


def shuffle(data,labels=None):

    if labels is not None:
        assert len(data) == len(labels)

    for i in range(0,len(data)):
        rindex = np.random.randint(0,len(data))
        tmpx = data[rindex]
        data[rindex] = data[i]
        data[i] = tmpx

        if labels is not None:
            tmpy = labels[rindex]
            labels[rindex] = labels[i]
            labels[i] = tmpy

    if labels is None:
        return data

    return data,labels


def len_deepest(ar):
    x = np.array(ar).tolist()
    assert type(x) == list

    while type(np.array(x[0]).tolist()) == list:
        x = np.array(x[0]).tolist()

    return len(x)


def select(ar, val, op = '==',column = None):
    aprime = []

    if column is not None:
        if (op == '=='):
            aprime = [element for element in ar if element[column] == val]
        elif (op == '<='):
            aprime = [element for element in ar if element[column] <= val]
        elif (op == '>='):
            aprime = [element for element in ar if element[column] >= val]
        elif (op == '<'):
            aprime = [element for element in ar if element[column] < val]
        elif (op == '>'):
            aprime = [element for element in ar if element[column] > val]
        elif (op == '!='):
            aprime = [element for element in ar if element[column] != val]
        elif (op == 'IN' or op == 'in'):
            aprime = []
            if type(val) is list:
                for i in val:
                    [aprime.append(element) for element in ar if element[column] == i]
            else:
                [aprime.append(element) for element in ar if element[column] == val]
        else:
            print 'Unknown Operation'
    else:
        if (op == '=='):
            aprime = [element for element in ar if element == val]
        elif (op == '<='):
            aprime = [element for element in ar if element <= val]
        elif (op == '>='):
            aprime = [element for element in ar if element >= val]
        elif (op == '<'):
            aprime = [element for element in ar if element < val]
        elif (op == '>'):
            aprime = [element for element in ar if element > val]
        elif (op == '!='):
            aprime = [element for element in ar if element != val]
        else:
            print 'Unknown Operation'
    return aprime


def nanlen(ar):
    x = 0
    for i in range(0,len(ar)):
        if not np.isnan(ar[i]):
            x += 1
    return x


def numerate(ar, ignore=list([])):
    ar = list(ar)
    assert type(ignore) is list

    u = unique(ar, ignore)

    for i in range(0,len(ar)):
        found = False
        for j in range(0,len(u)):
            if ar[i] == u[j]:
                ar[i] = j
                found = True
                break
        if not found:
            ar[i] = float('nan')
    return ar


def transpose(ar):
    L = len(ar)
    if isinstance(ar[0],float):
        for i in range(0,L):
            ar[i] = [ar[i]]
        W = 1
    else:
        W = len(ar[0])

    nArr = []

    for i in range(0, W):
        nrow = []
        for j in range(0,L):
            nrow.append(ar[j][i])
        nArr.append(nrow)

    return nArr



