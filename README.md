# modelpipeline


How to run
---------
Please found a sample run I did and useful note at [Note](https://github.com/kaiwang0112006/modelpipeline/blob/master/example/note)

##run from file
python run.py --train=trainhead.csv --test=testhead.csv --model=randomForest

##use as python api
    
    from run import *
    modelrun(trina,test,fts,model,ftscol,lstcol)
