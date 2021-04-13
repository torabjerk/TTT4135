#Task 1

import numpy as np

#data = np.loadtxt("data1.txt")
data = np.loadtxt("data2.txt")
#data3 = np.loadtxt("data3.txt")

data_len = np.size(data)
ones = 0
zerohs = 0

for i in range(np.size(data)):
    if data[i] == 1:
        ones = ones + 1
    else:
        zerohs = zerohs + 1

print("Data 2")
print("Ones ", ones)
print("Zerohs ", zerohs)
print("Size of data ", data_len)

p_ones = float(ones)/float(data_len)
p_zerohs = float(zerohs)/float(data_len)

print("P_Ones ", p_ones)
print("P_Zerohs ", p_zerohs)

H = -p_zerohs*np.log2(p_zerohs) - (p_ones)*np.log2(p_ones)
print("The entropy is for data2 is ", H)
