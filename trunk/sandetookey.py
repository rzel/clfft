import sys
import math

def sandetookey(n):
    lgn = math.log(n,2)

    for ildm in range(lgn):
        m = 2 ** ( lgn - ildm)
        mh = m/2
        iter =0
        for j in range(mh):
            
            for r in range(0,n-m + 1,m):
                iter += 1
                print iter,"r:",r,"j:",j,"r + j", r+j, "r + j + mh", r+j+mh
                

        print "-" * 30



sandetookey(int(sys.argv[1]))
