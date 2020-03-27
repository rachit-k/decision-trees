import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import sys
from xclib.data import data_utils
import timeit


def ENTROPY(Y):
    vals, counts = np.unique(Y, return_counts=True)
    t=Y.shape[0]
    entropy=0.0
    for count in counts:
        entropy=entropy-(count/t)*(math.log(count/t))
        
    return entropy    


def IG(X,Y,x):
    parentE= ENTROPY(Y)
    med=np.median(X[:,x], axis=0)
    t=X.shape[0]
#    print(med.shape)
    count0=0
    count1=0
    Y0=[]
    Y1=[]
    for i in range(t):
        if (X[i,x] <= med):
            count0=count0 + 1
            Y0.append(Y[i])
        else:
            count1=count1 + 1
            Y1.append(Y[i])
    Y0=np.array(Y0)
    Y1=np.array(Y1)
    childE=(count0*ENTROPY(Y0)+count1*ENTROPY(Y1))/t
    return (parentE-childE)

def recur(X,Y,depth):
#    print(depth)
    if (depth==0):
        return (['l',-1,0,-1,{}],1)
    if (len(X)==0):  
        return (['l',-1,0,-1,{}],1)
#all have same output
    elif (len(np.unique(Y))==1):
#        print('leaf0')
        return (['l',-1,Y[0],-1,{}],1)
#all have same input  
    elif ((np.unique(X,axis=1)).shape[1]==1): #check
#        print('leaf1')
        vals, counts = np.unique(Y, return_counts=True)    
        if (counts[0]>counts[1]):
            return (['l',-1,vals[0],-1,{}],1)
        else:
            return (['l',-1,vals[1],-1,{}],1)
    else:
#        tree=[typ,dividing feature,max,median,dict] ;typ=normal/leaf
#        print('enter')
        tree=['n',-1,-1,-1,{}]   
        nodes=1
        igs=[]
        x=X.shape[1]
        for i in range(x):
            igs.append(IG(X,Y,i))
        igs=np.array(igs)    
        maxig=np.max(igs)   
        maxig_ind=np.where(igs==maxig)
#        print(maxig_ind[0][0])
        maxig_ind=maxig_ind[0][0]
        med=np.median(X[:,maxig_ind], axis=0)
#        med=med[0][0]
        tree[1]=maxig_ind
        tree[3]=med
        maxval=0
        vals, counts = np.unique(Y, return_counts=True)
        if (counts[0]>counts[1]):
            maxval=vals[0]
        else:
           maxval=vals[1]         
        if (maxig<=0):
#            print('leaf')
            return (['l',-1,maxval,-1,{}],nodes)             
        tree[2]=maxval
        
        new_x = []
        for i in range(x):
            if (i != maxig_ind):
                new_x.append(i)
        X0=[]
        X1=[]
        Y0=[]
        Y1=[]
        t=len(Y)
        for i in range(t):
            if (X[i,maxig_ind] <= med):
                X0.append(X[i])
                Y0.append(Y[i])
            else:
                X1.append(X[i])     
                Y1.append(Y[i])   
        X0=np.array(X0)
#        print(X0.shape)
        X0=X0.reshape((X0.shape[0],x))
        X1=np.array(X1)
        X1=X1.reshape((X1.shape[0],x))
        Y0=np.array(Y0)
        Y1=np.array(Y1) 
#        print("X0.shape")
#        print(X0.shape)
#        print(X0)
#        print(X0[2,4])
        retval0=recur(X0,Y0,depth-1)
        retval1=recur(X1,Y1,depth-1)
        tree[4][0]=retval0[0]
        tree[4][1]=retval1[0]
        nodes=nodes+retval0[1]+retval1[1]
#        print('normal')
        return (tree,nodes) 
    


    
def predict_help(X,tree):
    if(tree[0]=='l'):
        return tree[2]
    pred=1
    if(X[tree[1]]<=tree[3]):
        pred=0
    if (pred in tree[4].keys()):  
        return predict_help(X,tree[4][pred])
    return tree[2]
    
def predict(X,Y,tree):
    prediction=[]
    for i in range(len(Y)):
        prediction.append(predict_help(X[i,:].reshape((X.shape[1],1)), tree))
    prediction=np.array(prediction)
    return (1-(np.count_nonzero(prediction-Y)/len(Y)))



def prune_help(vX,vY,tree):
    if (tree[0] == 'leaf'):
        return(-1, tree)
    acc=predict(vX,vY,tree)
    tree1=tree.copy()
    for i in tree1[4].keys():
#        print(i)
        new_acc,new_tree=prune_help(vX,vY,tree1[4][i])
        if (new_acc>acc):
            acc=new_acc
            tree1[4][i]=new_tree
    return (acc,tree1)


#def prune(X,Y,vX,vY,tX,tY,tree):
#    acc_train_list=[]
#    acc_valid_list=[]
#    acc_test_list=[]
#    tree1=tree.copy()
#    while(True):
#        acc_train=predict(X,Y,tree1)
#        acc_valid=predict(vX,vY,tree1)
#        acc_test=predict(vX,vY,tree1)
#        acc_train_list.append(acc_train)
#        acc_valid_list.append(acc_valid)
#        acc_test_list.append(acc_test)
#        new_acc,new_tree=prune_help(vX,vY,tree1)
#        if (acc_valid>new_acc):
##            print('stop pruning')
#            break
#
#    return (acc_train_list,acc_valid_list,acc_test_list) 
def prune(X,Y,vX,vY,tX,tY,tree):
    tree1=tree.copy()
    while(True):
        acc_valid=predict(vX,vY,tree1)
        new_acc,new_tree=prune_help(vX,vY,tree1)
        if (acc_valid>=new_acc):   
            acc_train=predict(X,Y,new_tree)
            acc_test=predict(tX,tY,new_tree)
            print('stop pruning')
            break
        tree1=new_tree

    return (acc_train,acc_valid,acc_test) 



train_x = sys.argv[1]
train_y = sys.argv[2]
valid_x = sys.argv[3]
valid_y = sys.argv[4]
test_x = sys.argv[5]
test_y = sys.argv[6]

    
starttime = timeit.default_timer()    

X = data_utils.read_sparse_file(train_x, force_header=True)
X=X.todense() 
with open(train_y) as f1:
    Y = np.genfromtxt(f1)
    
vX = data_utils.read_sparse_file(valid_x, force_header=True)
vX=vX.todense()
with open(valid_y) as f1:
    vY = np.genfromtxt(f1)
  
tX = data_utils.read_sparse_file(test_x, force_header=True)
tX=tX.todense()
with open(test_y) as f1:
    tY = np.genfromtxt(f1)
    
#print("time:")    
#print(timeit.default_timer()-starttime)
depths=[10,20,25,30,35,40,45,50,52]

train_list=[]
test_list=[]
valid_list=[]
nodes_list=[]
tree_list=[]
for depth in depths:
    tree,nodes=recur(X,Y,depth)
    
    nodes_list.append(nodes)
    tree_list.append(tree)
    print("validation accuracy:")    
    accuracy =predict(X,Y,tree)
    print(accuracy)
    train_list.append(accuracy)
    
    print("validation accuracy:")    
    accuracy =predict(vX,vY,tree)
    print(accuracy)
    valid_list.append(accuracy)
    
    print("test accuracy")    
    accuracy = predict(tX,tY,tree)
    print(accuracy)
    test_list.append(accuracy)


plt.plot(nodes_list,train_list,'-o',label='train',color='red',)
plt.plot(nodes_list,valid_list,'-o',label='validation',color='blue')
plt.plot(nodes_list,test_list,'-o',label='test',color='green',)
plt.xlabel('number of nodes')
plt.ylabel('accuracy')
plt.legend()
plt.savefig("Q1.png",bbox_inches="tight")
plt.show()  

#print("time:")
#print(timeit.default_timer()-starttime)

train_list1=[]
test_list1=[]
valid_list1=[]

#starttime = timeit.default_timer()    
#
for tree in tree_list:
    acc_train,acc_valid,acc_test=prune(X,Y,vX,vY,tX,tY,tree)
    train_list1.append(acc_train)
    valid_list1.append(acc_valid)
    test_list1.append(acc_test)
    
#
#print("time:")    
#
#print(timeit.default_timer()-starttime)

plt.plot(nodes_list,train_list1,'-o',label='train',color='red',)
plt.plot(nodes_list,valid_list1,'-o',label='validation',color='blue')
plt.plot(nodes_list,test_list1,'-o',label='test',color='green')
plt.xlabel('number of nodes')
plt.ylabel('accuracy')
plt.legend()
#plt.savefig("Q2.png",bbox_inches="tight")
plt.show()
