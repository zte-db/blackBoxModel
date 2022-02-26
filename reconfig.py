import sys

def replace(tps,fileName="/home/zte/oltpbench/config/pgtpcc_config.xml"):
    with open(fileName,"r") as f:
        lines = f.readlines()
        lines[14]="<rate>"+tps+"</rate>\n"
        f.close()
    
    with open(fileName,"w") as f:
        for line in lines:
            f.write(line)
        f.close()


if __name__=='__main__':
    replace(sys.argv[1])