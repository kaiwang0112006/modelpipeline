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
    requiredgroup.add_argument('--train',dest='train',help='train', default='')
    requiredgroup.add_argument('--test',dest='test',help='test', default='')
    #requiredgroup.add_argument('--conf',dest='configuration file',help='conf', default='')
    parser.add_argument('--ftscol',dest='ftscol',help='whether the first column is the id that will not be treat as id, true or false', default='true')
    parser.add_argument('--lstcol',dest='lstcol',help='whether the last column of test data is a target column, true or false', default='false')
    parser.add_argument('--fts',dest='fts',help='feture selection et. extraTrees, cor, randomTree. none', default='none')
    parser.add_argument('--model',dest='model',help="Classifiers, {'gBoosting', 'randomForest'}", default='gBoosting')

    args = parser.parse_args()

    return args

def modelrun(trina,test,fts,model,ftscol='true',lstcol='false'):
    print 'reading file'
    parseClass = parseData(trina,test)
    parseClass.parseRead(ftscol,lstcol)
    print 'process data'
    modelClass = domodel(parseClass.train_x, parseClass.train_y, parseClass.test_x, parseClass.test_y)
    modelClass.processData()
    print 'featureSelect'
    modelClass.featureSelect(fts)
    print 'modeling...'
    if model == 'gBoosting':
        model = modelClass.doGradientBoostingClassifier()
        best_parameters = model.best_estimator_.get_params()
        for param_name in sorted(best_parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    elif model == 'randomForest':
        model = modelClass.doRandomforest()
        best_parameters = model.best_estimator_.get_params()
        for param_name in sorted(best_parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    
    

##########################################
## Master function
##########################################           
def main():
    options = getOptions()
    modelrun(options.train, options.test,options.fts,options.model,ftscol=options.ftscol,lstcol=options.lstcol)
    
    
    
    
        
    
if __name__ == "__main__":
    main()