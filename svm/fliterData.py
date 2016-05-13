'''
Filte  the data for matlab to plot the data set
the data file is from libsvm: fourcalss_scale
'''

X= []
y= []

fobj = open('fourclass_scale','r')
for line in fobj:

    tmp = line.strip().split(' ')
    if len(tmp) < 3: continue
    y.append(int(tmp[0]))
    X.append([float(tmp[1].split(':')[1]), float(tmp[2].split(':')[1])])

fobj.close()

assert len(y) == len(X)

fobj = open('filtered.txt','w')

for i in range(0, len(y)):
    fobj.write(str(y[i]))
    fobj.write(' ')
    fobj.write(str(X[i][0]))
    fobj.write(' ')
    fobj.write(str(X[i][1]))
    fobj.write('\n')

fobj.close()