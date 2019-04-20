import pandas as pd
import numpy as np
import json
from math import log2
from dataset import examples, target_attr, attr_list

# Target attribute
t_attr = 'play'

def is_positive_examples(examples):
    target = pd.Series(pd.Categorical(examples[t_attr], categories=['no','yes']))
    if target.value_counts()[1] == target.count():
        return True
    else:
        return False

def is_negative_examples(examples):
    target = pd.Series(pd.Categorical(examples[t_attr], categories=['no','yes']))
    if target.value_counts()[0] == target.count():
        return True
    else:
        return False

def is_empty_attr_list(attr_list):
    if attr_list:
        return False
    else:
        return True

def most_common(examples):
    target = pd.Series(pd.Categorical(examples[t_attr], categories=['no','yes']))
    if target.value_counts()[1] >= target.value_counts()[0]:
        return target.cat.categories[1]
    else:
        return target.cat.categories[0]

# Entropy formula
def entropy(matrix, target_values, t_target_values, n):
    entropy = []
    entropy_of_i = 0
    entropy_value = 0
    for i in range(0,n):
        for j in range(0,2):
            if matrix[j,i] != 0:
                entropy_of_i += -(matrix[j,i]/target_values[i])*log2(matrix[j,i]/target_values[i])
            else:
                entropy_of_i += 0
        entropy.append(entropy_of_i)
        entropy_of_i = 0    
    for i in range(0, len(entropy)):
        entropy_value += (target_values[i]/len(examples))*entropy[i]
    return entropy_value    

def entopy_parent(t_target_values, length):
    entropy=0
    for i in range(0, 2):
        if t_target_values[i]:
            entropy += -(t_target_values[i]/length)*log2(t_target_values[i]/length)
        else:
            entropy += 0
    return entropy

# Entropy calculation
def info_gain(examples, attr, target, target_attr):
    t_target = pd.Series(pd.Categorical(examples[t_attr], categories=['no','yes']))
    matrix = np.zeros((2, target.nunique()))
    for i in range(0, target.nunique()):
        for j in range(0, 2):
            matrix[j][i] = len(examples[((examples[attr] == target.cat.categories[i]) & (examples[t_attr] == t_target.cat.categories[j]))].index)
    t_target_values = t_target.value_counts().tolist()
    target_values = target.value_counts().tolist()
    t_target_values.reverse()
    target_values.reverse()
    parent_entropy = entopy_parent(t_target_values, len(examples))
    entropy_value = entropy(matrix, target_values, t_target_values, target.nunique())
    return parent_entropy - entropy_value

# Find the best split using entropy
def best_split(examples, target_attr, attr_list):
    info_gain_values = []
    for i in attr_list:
        target = pd.Series(pd.Categorical(examples[i]))
        info_gain_i = info_gain(examples, i, target, target_attr)
        info_gain_values.append(info_gain_i)
    max = 0
    for i in range(1,len(info_gain_values)):
        if info_gain_values[max] < info_gain_values[i]:
            max = i
    return examples.columns[max]

def get_values(A):
    target = pd.Series(pd.Categorical(examples[A]))
    return target.cat.categories.tolist()

# Decision Tree Algorithm
def ID3(examples, target_attr, attr_list):
    "ID3 Decision Tree Algorithm"
    if is_positive_examples(examples):
        temp_root = 'yes'
    if is_negative_examples(examples):
        temp_root = 'no'
    if is_empty_attr_list(attr_list):
        temp_root = most_common(examples)
    else:
        A = best_split(examples, target_attr, attr_list)
        values = get_values(A)
        temp_root = {}
        for i in values:
            examples_i = examples[examples[A] == i]
            if examples_i.empty:
                temp_root[str(i)] = most_common(examples)
            else:
                temp_root[str(i)] = ID3(examples_i.drop([A], axis = 1), target_attr, [i for i in attr_list if i != A])
    return temp_root
    
root = ID3(examples, target_attr, attr_list)
print ("Decision Tree a dictionary data structure")
print (json.dumps(root, indent=8))
print ("\n")

# Predict Function
def predict(root, test):
    keys = list(root.keys())
    index = 0
    for i in keys:
        for j in test:
            if i == j:
                index = i
    value = root.get(index)
    if isinstance(value, dict):
        test.remove(index)
        return predict(value, test)
    else:
        return value 

def predict_list(root, test):
    list = []
    for i in test:
        list.append(predict(root,i))
    return list

test = [['sunny','cool','high', 'False'],['sunny','cool','high','True'],['overcast','hot','high','True']]
print ("Test data")
print (test)
result = predict_list(root, test)
print ('Prediction: ',result)