#Task 1

import numpy as np

data = np.loadtxt("data1.txt")
#data2 = np.loadtxt("data2.txt")
#data3 = np.loadtxt("data3.txt")

## Task 1d
data_len = np.size(data)
ones = 0
zerohs = 0

for i in range(data_len):
    if data[i] == 1:
        ones = ones + 1
    else:
        zerohs = zerohs + 1

print("Data 1")
print("Ones ", ones)
print("Zerohs ", zerohs)
print("Size of data ", data_len)

p_ones = float(ones)/float(data_len)
p_zerohs = float(zerohs)/float(data_len)

print("P_Ones ", p_ones)
print("P_Zerohs ", p_zerohs)

H = -p_zerohs*np.log2(p_zerohs) - (p_ones)*np.log2(p_ones)
print("The entropy is for data1 is ", H)

## Task 1e
data_2bit = []
oneone = 0
onezero = 0
zeroone = 0
zerozero = 0

for i in range(0,data_len,2):
    if data[i] == 1:
        if data[i+1] == 1:
            oneone += 1
        elif data[i+1] == 0:
            onezero += 1
    else:
        if data[i+1] == 1:
            zeroone += 1
        elif data[i+1] == 0:
            zerozero += 1

print(f"Number of 11s:{oneone}")
print(f"Number of 10s: {onezero}")
print(f"Number of 01s:{zeroone}")
print(f"Number of 00s:{zerozero}")
