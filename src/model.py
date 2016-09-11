from featureSel import *
import numpy as np
from sklearn import grid_search
from sklearn.ensemble import *
from sklearn.feature_selection import VarianceThreshold
from smote import *
from numpy.distutils.__version__ import minor

class domodel(object):
    def __init__(self,train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        
    def processData(self):
        #remove feature with no distinction and less important
        print "remove feature with no distinction and less important"
        indices = [i for i in range(len(self.train_x[0]))]
        frqIndex = trimfrq(self.train_x)
        
        for i in frqIndex:
            indices.remove(i)
        train_x_uniq = indexTodata(self.train_x, indices)
        test_x_uniq = indexTodata(self.test_x, indices)
        
        #normalization
        print "normalization"
        train_x_nor, mean, std = normalize(train_x_uniq)
        test_x_nor, mean, std = normalize(test_x_uniq, mean, std)
        
        self.train_x = train_x_nor
        self.test_x = test_x_nor
        
    def smoteRun(self,N,k):
        yset = list(set(self.train_y))
        ydict = {yset[0]:[],yset[1]:[]}
        
        for index,x in enumerate(self.train_x):
            ydict[self.train_y[index]].append(x)
        
        if len(ydict[yset[0]]) >= len(ydict[yset[1]]):
            synsamps = SMOTE(np.asarray(ydict[yset[1]]), N, k)
            self.train_x = ydict[yset[0]] + synsamps.tolist()
            minorityIndex = [yset[1],len(synsamps)]
            majorityIndex = [yset[0],len(ydict[yset[0]])]
        else:
            synsamps = SMOTE(np.asarray(ydict[yset[0]]), N, k)
            self.train_x = ydict[yset[1]] + synsamps.tolist()   
            minorityIndex = [yset[0],len(synsamps)]
            majorityIndex = [yset[1],len(ydict[yset[1]])]

        y = [] 
        for i in range(majorityIndex[1]):
            y.append(majorityIndex[0])
        for i in range(minorityIndex[1]):
            y.append(minorityIndex[0])
            
        self.train_y = y
            
        
        
    def featureSelect(self,method=None,k=None,fun=None):
        if method == 'cor':
            train_x_sel, test_x_sel = correlationSelect(train_x_nor, self.train_y, self.test_x_nor)
        elif method == 'extraTrees':
            train_x_sel, test_x_sel = ExtraTreesSelect(train_x_nor, self.train_y, self.test_x_nor)
        elif method == 'randomTree':
            train_x_sel, test_x_sel = randomTreesSelect(train_x_nor, self.train_y, self.test_x_nor)
        elif method == 'kbest':
            train_x_sel, test_x_sel = SelectKb(train_x_nor, self.train_y, self.test_x_nor,k=k,fun=fun)
        else:
            return
        self.train_x = train_x_sel
        self.test_x = test_x_sel 
        
    def doGradientBoostingClassifier(self):
        parameters = {'loss':('deviance', 'exponential'),
                      'learning_rate':np.arange(0.1,0.9,0.5),
                      'n_estimators':range(100,120,10),
                      'max_depth':[3,4,5],
                      }

        clf = grid_search.GridSearchCV(GradientBoostingClassifier(), parameters)
        clf.fit(self.train_x, self.train_y)
        return clf
    
    def doRandomforest(self):
        parameters = { 
            'n_estimators': range(100,120,10),
            'max_features': ['auto', 'sqrt', 'log2',None]
        }
        
        clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
        clf.fit(self.train_x, self.train_y)
        return clf
        
def trimfrq(des):
    desmap = map(list, zip(*des))
    desindex = []
    for index,eachdes in enumerate(desmap):
        eachdes = list(set(eachdes))
        if len(eachdes) == 1:
            desindex.append(index)
    return desindex

def indexTodata(data, indices):
    newdata = []
    for eachline in data:
        temp = []
        for i in indices:
            try:
                temp.append(eachline[i])
            except:
                print (len(eachline))
                exit(0)
        newdata.append(temp)
        
    return newdata       

def normalize(x, mean=None, std=None):
    flist = map(list, zip(*x))
    
    count = len(x)
    if mean is None:
        mean = []
        for i in flist:
            mean.append(np.mean(i))
    if std is None:
        std = []
        for i in flist:
            std.append(np.std(i))
    for i in range(count):
        for j in range(len(x[i])):
            x[i][j] = (x[i][j]-mean[j])/std[j]
    return x, mean, std