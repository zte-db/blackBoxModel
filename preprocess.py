import sys

def readData(fileName="./sql.log"):
    with open(fileName,"r") as f:
        data=[]
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(": ")
            tmp=[]
            tmp.append(getTimeStamp(line[0].split(".")[0]))
            tmp.append(line[-1])
            data.append(tmp)
        f.close()

        return data

def getTimeStamp(date):
    import time

    #dt = "2016-05-05 20:28:54"
    #转换成时间数组
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    #转换成时间戳
    timestamp = time.mktime(timeArray)
    return timestamp

def WriteToFile(data,fileName):
    with open("./data/data" + fileName + ".txt","w") as f:
        for line in data:
            f.write(str(line[0]).split(".")[0] + "\t\t" + line[1] + "\n")
        f.close()

if __name__=='__main__':
    data = readData()
    WriteToFile(data,sys.argv[1])

