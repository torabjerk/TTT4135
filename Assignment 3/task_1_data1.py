#Task 1

import numpy as np

data = np.loadtxt("data1.txt")
#data2 = np.loadtxt("data2.txt")
#data3 = np.loadtxt("data3.txt")
content = open('data1.txt','r').read().replace('\n','')

## Task 1d
data_len = np.size(data)
ones = 0
zerohs = 0

for i in range(data_len):
    if data[i] == 1:
        ones = ones + 1
    else:
        zerohs = zerohs + 1

print("\n")
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
print("\n")

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

print(f"Number of 11s: {oneone}")
print(f"Number of 10s: {onezero}")
print(f"Number of 01s: {zeroone}")
print(f"Number of 00s: {zerozero}")
print("\n")

p_oneone = float(oneone)/float(data_len)
p_onezero = float(onezero)/float(data_len)
p_zeroone = float(zeroone)/float(data_len)
p_zerozero = float(zerozero)/float(data_len)

H_2bit = -p_zerozero*np.log2(p_zerozero) - (p_zeroone)*np.log2(p_zeroone) \
- (p_onezero)*np.log2(p_onezero) -(p_oneone)*np.log2(p_oneone)
print(f"The entropy of two bits is {H_2bit}")

n3=3
data_3bit =[content[i:i+n3] for i in range(0,len(content),n3)]

occurence_list_3bit = ["000", "001", "010", "011", "100", "101", "110", \
"111"]

dict_3bit = {"000": 0,
 "001": 0,
 "010": 0,
 "011": 0,
 "100": 0,
 "101": 0,
 "110": 0,
 "111": 0,
}

H_3bit = 0

for i in range(0, np.size(data_3bit)):
    key = data_3bit[i]
    if key in occurence_list_3bit:
        dict_3bit[key] += 1

for key in dict_3bit:
    H_3bit -= dict_3bit[key]/np.size(data_3bit)*np.log2(dict_3bit[key]/np.size(data_3bit))

print(f"The entropy of three bits: {H_3bit}")

n=4
data_4bit =[content[i:i+n] for i in range(0,len(content),n)]

occurence_list_4bit = ["0000", "0001", "0010", "0011", "0100", "0101", "0110", \
"0111", "1000", "1001", "1010","1011", "1100", "1101", "1110", "1111"  ]

dict_4bit = {"0000": 0,
 "0001": 0,
 "0010": 0,
 "0011": 0,
 "0100": 0,
 "0101": 0,
 "0110": 0,
 "0111": 0,
 "1000": 0,
 "1001": 0,
 "1010": 0,
 "1011": 0,
 "1100": 0,
 "1101": 0,
 "1110": 0,
 "1111": 0
}

dict_4bit_p = {"0000": 0,
 "0001": 0,
 "0010": 0,
 "0011": 0,
 "0100": 0,
 "0101": 0,
 "0110": 0,
 "0111": 0,
 "1000": 0,
 "1001": 0,
 "1010": 0,
 "1011": 0,
 "1100": 0,
 "1101": 0,
 "1110": 0,
 "1111": 0
}

H_4bit = 0

for i in range(0, np.size(data_4bit)):
    key = data_4bit[i]
    if key in occurence_list_4bit:
        dict_4bit[key] += 1

print("\n")
print(f"Number of {dict_4bit}")

for key in dict_4bit:
        dict_4bit_p[key] = dict_4bit[key]/np.size(data_4bit)

print("\n")
print(f"Number of {dict_4bit_p}")
print("\n")

for key in dict_4bit:
    H_4bit -= dict_4bit[key]/np.size(data_4bit)*np.log2(dict_4bit[key]/np.size(data_4bit))

print(f"The entropy of four bits: {H_4bit}")
