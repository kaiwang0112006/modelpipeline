import argparse
from parse import *
from model import *
from featureSel import *
from sklearn.ensemble import *
from sklearn.feature_selection import VarianceThreshold

##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option]"',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    requiredgroup = parser.add_argument_group('required arguments')
    requiredgroup.add_argument('--train',dest='train',help='train', default='', required=True)
    requiredgroup.add_argument('--test',dest='test',help='test', default='', required=True)
    #requiredgroup.add_argument('--conf',dest='configuration file',help='conf', default='')
    parser.add_argument('--ftscol',dest='ftscol',help='whether the first column is the id that will not be treat as id, true or false', action='store_true', default=False)
    parser.add_argument('--lstcol',dest='lstcol',help='whether the last column of test data is a target column, true or false', action='store_true', default=False)
    parser.add_argument('--fts',dest='fts',help='feture selection et. extraTrees, cor, randomTree. none', default='none')
    parser.add_argument('--model',dest='model',help="Classifiers, {'gBoosting', 'randomForest'}", default='gBoosting')
    parser.add_argument('--smote', dest='smote',help="whether or not use the SMOTE Algorithm",action='store_true', default=False)
    parser.add_argument('--synsamp',dest='synsamp',help='percetange of new synthetic samples.(int,SMOTE Algorithm Parameter)', type=int, default=100)
    parser.add_argument('--nbs',dest='nbs',help='Number of nearest neighbours.(int,SMOTE Algorithm Parameter)', type=int, default=1)
    args = parser.parse_args()

    return args

def modelrun(train,test,fts,modelsel,ftscol=False,lstcol=False,smote=[]):
    print 'reading file'
    parseClass = parseData(train,test)
    parseClass.parseRead(ftscol,lstcol)
    print 'process data'
    modelClass = domodel(parseClass.train_x, parseClass.train_y, parseClass.test_x, parseClass.test_y)
    modelClass.processData()
    if smote != []:
        print 'smote'
        modelClass.smoteRun(smote[0], smote[1])
    print 'featureSelect'
    modelClass.featureSelect(fts)
    print 'modeling...'
    if modelsel == 'gBoosting':
        model = modelClass.doGradientBoostingClassifier()
        best_parameters = model.best_estimator_.get_params()
        for param_name in sorted(best_parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    elif modelsel == 'randomForest':
        model = modelClass.doRandomforest()
        best_parameters = model.best_estimator_.get_params()
        for param_name in sorted(best_parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    
    

##########################################
## Master function
##########################################           
def main():
    options = getOptions()
    if options.smote:
        smote = [options.synsamp, options.nbs]
    else:
        smote = []
    modelrun(options.train, options.test,options.fts,options.model,ftscol=options.ftscol,lstcol=options.lstcol,smote=smote)
    
    
        
    
if __name__ == "__main__":
    main()