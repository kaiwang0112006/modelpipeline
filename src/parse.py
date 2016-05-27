import csv

class parseData(object):
    def __init__(self,trainfile,testfile):
        self.trainfile = trainfile
        self.testfile = testfile
        self.nonNumcontent = {}
        
    def parseRead(self,fstcol=False,lstcol=False):
        self.train_x, self.train_y= self.readcsv(self.trainfile,fstcol,True)
        self.test_x, self.test_y = self.readcsv(self.testfile,fstcol,lstcol)
        
        
    def readcsv(self,file,fscol,lstcol='true'):
        x = []
        y = []
        if fscol:
            cst = 1
        else:
            cst = 0
        csvfile = open(file)
        fin = csv.reader(csvfile)
        
        for eachline in fin:
            if (fin.line_num == 1): 
                continue
    #         if fin.line_num == lineNo:
    #             break

            if lstcol:
                x.append([self.judegvalue(cell,index) for index,cell in enumerate(eachline[cst:-1])])
                y.append(float(eachline[-1]))
            else:
                x.append([self.judegvalue(cell,index) for index,cell in enumerate(eachline[cst:])])
                y.append(1.0)
    
        csvfile.close()
        return (x,y)
            
    def judegvalue(self,cell,i):
        if "\"" in cell:
            cell = cell.strip()[1:-1]
        try:
            return float(cell)
        except:
            if cell.upper() == 'FALSE' or cell == "NA" or cell == "" or cell == "[]":
                return 0
            elif cell.upper() == 'TRUE':
                return 1
            else:
                if not i in self.nonNumcontent:
                    self.nonNumcontent[i] = {cell:1}
                    return 1
                else:
                    if cell in self.nonNumcontent[i]:
                        return self.nonNumcontent[i][cell]
                    else:
                        maxval = max(self.nonNumcontent[i].values())
                        self.nonNumcontent[i][cell] = maxval + 1
                return maxval + 1
        