import pandas as pd

examples = pd.read_csv("train_data.csv")
target_attr = examples['play']
attr_list = list(examples)
attr_list.remove('play')

print ("Data set")
print (examples)
print ("\n")



