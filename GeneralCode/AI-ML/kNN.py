
####################################################
# TEAM MEMBERS BENJAMIN MONICAL AND JULIE WHITMORE #
####################################################

import random
import numpy as np
import math

####################################################
# kd tree                                          #
####################################################
def isNaN(x):
    return(x != x)

def medium(x):
    minV = 1000
    maxV = 0
    for i in x:
        if i < minV:
            minV = i
        if i > maxV:
            maxV = i
    return (minV + maxV)/2

def kd(x, y, column, min_points=40):

    x=np.array(x)
    y=np.array(y)
    kd_tree = dict()
    returns = [None]*2

    
    if (len(x) <= min_points or x.ndim < 2):
        returns[0] = x
        returns[1] = y
        return returns
    
    
    dimension = column%(x.shape[1]-1)
    value = medium(x[:, dimension])

    
    if np.all(x[:, dimension] == value):
        returns[0] = x
        returns[1] = y
        return returns
    
    x1 = [] 
    x0 = []
    y1 = []
    y0 = []
    
    for i in range(len(x)):
        if x[i, dimension] >= value:
            if len(x1) == 0:
                x1 = x[i]
                y1 = y[i]
            else:
                x1 = np.vstack((x1, x[i]))
                y1 = np.vstack((y1, y[i]))
        else:
            if len(x0) == 0:
                x0 = x[i]
                y0 = y[i]
            else:
                x0 = np.vstack((x0, x[i]))
                y0 = np.vstack((y0, y[i]))

 
    kd_tree[(dimension, value, True)] = kd(x1, y1, (dimension + 1), min_points)
    kd_tree[(dimension, value, False)] = kd(x0, y0, (dimension + 1), min_points)
    return kd_tree

def get_label(x, tree, neigh):
    
    label0 = 0
    label1 = 0
    knearest = predict_example(x, tree, neigh, [])

    for x in knearest:
        if(x[1] == 0):
            label0 += 1
        else:
            label1 += 1
            
    if label0 > label1:
        return 0
    else:
        return 1
    
def predict_example(x, tree, neigh, knearest):

    keys = []
    
    for key in tree.keys():
        keys.append(key)
    
    T = keys[0]
    F = keys[1]
   
    
    #TRUE BRANCH
    if (x[T[0]] >= T[1]): #T[0] = splitkey // T[1] = splitvalue
        if not isinstance(tree.get(T), dict):
            #LEAF
            points = tree.get(T) #returns [x, y]
            return nearest_neighbor(x, points[0], points[1], neigh, knearest)

        else:
            #NOT A LEAF, DESCEND TREE  
            new_knearest = predict_example(x, tree[T], neigh, knearest)
            
            #check to see if a point could be on other side of the border
            if(len(new_knearest) < neigh  or new_knearest[0][0] >= (F[1] - x[F[0]])**2 ):
            #Your other child is a leaf
                if not isinstance(tree.get(F), dict):
                    points = tree.get(F) 
                    return nearest_neighbor(x, points[0], points[1], neigh, new_knearest)
            #check other branch
                else:
                    return (predict_example(x, tree[F], neigh, new_knearest))
            else:
                return new_knearest

    #FALSE BRANCH
    if x[F[0]] < F[1] or isNaN(x[F[0]]):
        if not isinstance(tree.get(F), dict):
            #YOU ARE AT A LEAF
            points = tree.get(F) #returns [x, y]
            return nearest_neighbor(x, points[0], points[1], neigh, knearest)


        else:
            #NOT A LEAF DESCEND TREE    
            new_knearest = predict_example(x, tree[F], neigh, knearest)
            #Check to see if you should go to other side of the border  
            if(len(new_knearest) < neigh or  isNaN(x[F[0]]) or new_knearest[0][0] > (x[F[0]] - T[1])**2 ):
                 #Your other child is a leaf
                if not isinstance(tree.get(T), dict):
                    points = tree.get(T) #returns [x, y]
                    return nearest_neighbor(x, points[0], points[1], neigh, new_knearest)
                #Check other branch
                else:
                    return predict_example(x, tree[T], neigh, new_knearest)
                    
            else:
                return new_knearest

def distance(x1, x2):
    d = 0.0
    for i in range(len(x1)):
        if (not isNaN(x1[i]) and not isNaN(x2[i])):
            d += (x1[i]-x2[i])**2
    return d

def nearest_neighbor(xtest, xtrain, ytrain, k, knearest = []):

    count = 0

    #EMPTY XTRAIN
    if (xtrain.size == 0):
        return knearest

    #ONE XTRAIN
    if (xtrain.ndim < 2):
        dist = distance(xtest, xtrain)
        
        if(len(knearest) < k):
            knearest.append([dist, ytrain.item(count)])
            knearest.sort(reverse=True)
        else:
            if (knearest[0][0] > dist):
                knearest[0][0] = dist
                knearest[0][1] = ytrain.item(count)
                knearest.sort(reverse=True)
                
    #MANY XTRAIN
    else:
        for trainrow in xtrain:
            dist = distance(xtest, trainrow)

            if(len(knearest) < k):
                knearest.append([dist, ytrain.item(count)])
                knearest.sort(reverse=True)
            else:
                if (knearest[0][0] > dist):
                    knearest[0][0] = dist
                    knearest[0][1] = ytrain.item(count)
                    knearest.sort(reverse=True)
            count += 1

    return knearest

def visualize(tree, depth=0):

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}, {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            #print('+-- VALUES = {0}'.format(sub_trees))
            print('+-- VALUES')
            for x in sub_trees:
                print(x)

####################################################
# FEATURE SECLECTION kNN                           #
####################################################
def get_label_feature_selected(x, tree_list, neigh):
    x_new = x[tree_list[0]]
    label0 = 0
    label1 = 0
    knearest = predict_example(x_new, tree_list[1], neigh, [])

    for x in knearest:
        if(x[1] == 0):
            label0 += 1
        else:
            label1 += 1
            
    if label0 > label1:
        return 0
    else:
        return 1
    
def feature_selected_kd(x, y, num_features, algorithm):
    feature_tree = []
    features_to_use = []
    #get the results from an algorithm on which feature to use
    if(algorithm == "kendall"):
        features = kendall_best_features(x,y)
    else:
        features = spearmans_best_features(x,y)
        
    #put highest values upfront
    features.sort(reverse=True)
    for i in range(num_features):
        features_to_use.append(features[i][1])
        
    #partion data
    features_to_use.sort()
    feature_tree.append(features_to_use)
    
    x_new = x[:,features_to_use]
    
    tree = kd(x_new, y, 0)

    feature_tree.append(tree)

    return feature_tree
    


def rank(x_column):
    l = x_column.tolist()
    ranked = []
    i = 0
    j = 0
    smaller = 0
    equal = 0
    for i in range(len(l)):
        for j in range(len(l)):
            if( l[i] < l[j] and i != j):
                smaller += 1
            elif( l[i] == l[j] and i != j):
                equal += 1
        if not isNaN(l[i]):        
            ranked.append( smaller + .5*(equal-1))
            smaller = 0
            equal = 0
        else:
            ranked.append(float("nan"))
            smaller = 0
            equal = 0
    return ranked

####################################################
# Kendall Selected                                 #
####################################################

def kendall_best_features(x,y):
    result = []
    l = []
    x_columns = x.shape[1]            
    for i in range(x_columns):
                l.append(abs(kendall_correlation(x[: , i],y)))
                l.append(i)
                result.append(l)
                l = []
    return result
def kendall_correlation(x_column, y_column):
    x = rank(x_column)
    y = rank(y_column)
    length = len(x)
    concord = 0
    discord = 0
    for i in range(length):
        j = i + 1
        for j in range(length):
            if not isNaN(x[i]) and not isNaN(x[j]):
                if( x[i] < x[j] and y[i] < y[j] and i != j):
                    concord += 1
                elif( x[i] > x[j] and y[i] > y[j] and i != j ):
                    concord += 1
                elif( i != j):
                    discord += 1
    return (concord - discord)/ (concord + discord)

####################################################
# Spearman Selected                                #
####################################################
    
def spearmans_best_features(x,y):
    result = []
    l = []
    x_columns = x.shape[1]
    for i in range(x_columns):
                l.append(abs(spearmans_correlation(x[: , i],y)))
                l.append(i)
                result.append(l)
                l = [] 
    return result

def spearmans_correlation(x_column, y_column):
    x = rank(x_column)
    y = rank(y_column)
    length = len(x)
    x_mean = 0
    y_mean = 0
    top = 0
    bottom1 = 0
    bottom2 = 0

    #calculate mean of rank x
    for i in x:
        if not isNaN(i):
            x_mean = x_mean + i
    x_mean = x_mean/length
    
    #calculate mean of rank y
    for i in y:
        y_mean = y_mean + i
    y_mean = y_mean/length
    
    #spearman full formula
    for j in range(length):
        if not isNaN(x[j]):
            top = top + ( (x[j] - x_mean) * (y[j] - y_mean) )
            bottom1 = bottom1 + ( (x[j] - x_mean) **2)
            bottom2 = bottom2 + ((y[j] - y_mean) **2 )
        
    top = top/length
    bottom1 = bottom1/length
    bottom2 = bottom2/length
    return top/((bottom1*bottom2)**(1/2))        
    
####################################################
# BAGGED KNN                                       #
####################################################

def bagged_nearest_neighbor(x, y, num_trees):
    
    count = 0
    straps = bootstrap(x, y, num_trees)
    trees = [None]*num_trees
    for i in straps:
        trees[count] = [1, kd(i[0], i[1], 0)]
        count += 1
    return trees


def bootstrap(x,y, num_trees):
    xboot = [] 
    yboot = []
    onestrap = [None]*2
    allstrap = [None]*num_trees
    for j in range(num_trees): ### makes num_trees amount of bootstraps
        for i in range(len(y)): ### creates 1 bootstrap of length y
            z = random.randint(0,len(y)-1)
            if len(xboot) == 0:
                xboot = x[z]
                yboot = y[z]
            else:
                xboot = np.vstack((xboot, x[z]))
                yboot = np.vstack((yboot, y[z]))
        onestrap[0] = xboot
        onestrap[1] = yboot
        allstrap[j] = onestrap # save the bootstrap in an array of bootstraps
    #### reset the variables
        xboot = []
        yboot = []
        onestrap = [None]*2
    
    return allstrap


def predict_from_ensemble(x, h_ens, neigh):
    
    trues = 0
    falses = 0
    for tree in h_ens:
        if(get_label(x, tree[1], neigh) == 1):
            trues += tree[0]
        else:
            falses += tree[0]
            
    if(trues > falses):
        return 1
    else:
        return 0

####################################################
# RANDOM SUBSPACE KNN                              #
####################################################
def random_subspace(x, y, num_trees, num_subspace):
    
    count = 0
    straps = bootstrapfeature(x, y, num_trees, num_subspace)
    trees = [None]*num_trees

    for i in straps:
        trees[count] = [1 ,kd(i[0], i[1], 0), i[2]]
        count += 1
    return trees


def bootstrapfeature(x, y, num_trees, num_subspace):
    
    xboot = [] 
    yboot = []
    onestrap = [None]*3
    allstrap = [None]*num_trees
    subspace = []
    
    for j in range(num_trees): ### makes num_trees amount of bootstraps

        while len(subspace) != num_subspace:
            random_int = random.randint(0,Xtrn.shape[1]-1)
            if (random_int not in subspace):
                subspace.append(random_int)
            
        for i in range(len(y)): ### creates 1 bootstrap of length y
            z = random.randint(0,len(y)-1)
            if len(xboot) == 0:
                xboot = x[z]
                yboot = y[z]
            else:
                xboot = np.vstack((xboot, x[z]))
                yboot = np.vstack((yboot, y[z]))
        onestrap[0] = xboot[:, subspace]
        onestrap[1] = yboot
        onestrap[2] = subspace
        allstrap[j] = onestrap # save the bootstrap in an array of bootstraps
    #### reset the variables
        xboot = []
        yboot = []
        onestrap = [None]*3
        subspace = []
    
    return allstrap


def predict_from_subspace(x, h_ens, neigh):

    #0 = weight, 1 = tree, 2 = subspace

    trues = 0
    falses = 0
    for tree in h_ens:
        subspace = tree[2]
        if(get_label(x[subspace], tree[1], neigh) == 1):
            trues += tree[0]
        else:
            falses += tree[0]
            
    if(trues > falses):
        return 1
    else:
        return 0


####################################################
# Universal Stuff                                  #
####################################################

def confusion(y_true, y_pred):
    tp = 0 # true positive
    fp = 0 # false positive
    fn = 0 # false negative
    tn = 0 # true negative
    for i in range(len(y_pred)):
        if y_true[i] == 1 and  y_pred[i] == 1:
            tp +=1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn +=1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn +=1
        else:
            fp +=1
    print("True Positive = ", tp, " True Negative = ", tn, " False Positive = ", fp, " False Negative = ", fn)


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    bad = 0.0000
    for i in range(len(y_pred)):
        if (y_true[i] != y_pred[i]):
            bad +=1
    bad = (bad/len(y_pred))
    return bad

####################################################
# DATA MANIPULATION                                #
####################################################

def normalize(x):
    maxV = 0
    minV = 10000
    l = []
    if(np.all(x == x[0])):
        x.fill(1)
        return x
    
    for i in x:
        if( i < minV and not isNaN(i)):
            minV = i
        elif( i > maxV and not isNaN(i)):
            maxV = i
    for i in x:
        if(not isNaN(i)):
            i = (i- minV)/(maxV - minV)### Normalization formula x -xmin/ (xmax - xmin)
            l.append(i)
        else:
            l.append(i)
    return l



if __name__ == '__main__':
          
    # Load the training data
    M = np.genfromtxt('stroke-data25-75.train', missing_values=0, skip_header=0, delimiter=',', dtype=float)
    ytrn = M[:, 10]  # all of column 0
    Xtrn = M[:, 0:10] # all columns starting from column 1 (first column is excluded)

    # Load the test data
    M = np.genfromtxt('stroke-data25-75.test', missing_values=0, skip_header=0, delimiter=',', dtype=float)
    ytst = M[:, 10]
    Xtst = M[:, 0:10]
    
    Xtrn_columns = Xtrn.shape[1]
    for i in range(Xtrn_columns):
                Xtrn[:, i] = normalize(Xtrn[:, i])
       
    Xtst_columns = Xtst.shape[1]
    for i in range(Xtst_columns):
                Xtst[:, i] = normalize(Xtst[:, i])
  
    #############################
    # Regular kNN               #
    #############################
    
    
    #hyper parameter
    k = 1
    ################
    
    print("kNN Results ")
    
    kdtree = kd(Xtrn, ytrn, 0)
    
    #test error
    y_pred = [get_label(x, kdtree, k) for x in Xtst]
    confusion(ytst, y_pred)
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
    #training error
    y_pred = [get_label(x, kdtree, k) for x in Xtrn]
    tst_err = compute_error(ytrn, y_pred)
    print('Training Error = {0:4.2f}%.'.format(tst_err * 100))
    
    
    ################################
    # Spearman Feature Seclected kNN     
    ################################
    
    """
    #hyper parameter
    k = 1
    features = 3
    ################
    
    print("kNN Spearman selected features")
    kd_selected_feature = feature_selected_kd(Xtrn, ytrn, features, "spearman")

    y_pred = [get_label_feature_selected(x, kd_selected_feature, k) for x in Xtst]
    confusion(ytst, y_pred)
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
    #training error
    y_pred = [get_label_feature_selected(x, kd_selected_feature, k) for x in Xtrn]
    tst_err = compute_error(ytrn, y_pred)
    print('Training Error = {0:4.2f}%.'.format(tst_err * 100))
    """

    ################################
    # Kendall Feature Seclected kNN     
    ################################
    """

    #hyper parameter
    k = 1
    features = 5
    ################
    
    print("kNN Kendall selected features")
    kd_selected_feature = feature_selected_kd(Xtrn, ytrn, features, "kendall")

    y_pred = [get_label_feature_selected(x, kd_selected_feature, k) for x in Xtst]
    confusion(ytst, y_pred)
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
    #training error
    y_pred = [get_label_feature_selected(x, kd_selected_feature, k) for x in Xtrn]
    tst_err = compute_error(ytrn, y_pred)
    print('Training Error = {0:4.2f}%.'.format(tst_err * 100))
    """
    
    #############################
    # Bagged  kNN               #
    #############################
    
    
    """
    #hyper parameter
    k = 1
    bags = 10
    ################
    
    print("Bagged kNN Results Bags")

    bagged_kd = bagged_nearest_neighbor(Xtrn, ytrn, bags)  #num_trees

    #test error
    y_pred = [predict_from_ensemble(x, bagged_kd, k) for x in Xtst]
    confusion(ytst, y_pred)
    
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
    #training error
    y_pred = [predict_from_ensemble(x, bagged_kd, k) for x in Xtrn]
    tst_err = compute_error(ytrn, y_pred)
    print('Training Error = {0:4.2f}%.'.format(tst_err * 100))    
    """
    
    #############################
    # Random Subspace  kNN      #
    #############################
    """
    
    #hyper parameter
    k = 1
    bags = 30
    subspace = 3
    ################
    
    print("Random Subspace kNN Results Bags")
    
    trees = random_subspace(Xtrn, ytrn, bags, subspace)  #num_trees, num_subspace
    #test error
    y_pred = [predict_from_subspace(x, trees, k) for x in Xtst]
    confusion(ytst, y_pred)
    
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
    #training error
    y_pred = [predict_from_subspace(x, trees, k) for x in Xtrn]
    tst_err = compute_error(ytrn, y_pred)
    print('Training Error = {0:4.2f}%.'.format(tst_err * 100))
    """
