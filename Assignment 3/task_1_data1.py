#Task 1

import numpy as np
from sympy import symbols, limit

data = np.loadtxt("data3.txt")
#data2 = np.loadtxt("data2.txt")
#data3 = np.loadtxt("data3.txt")
content = open('data3.txt','r').read().replace('\n','')

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
print("Data 2")
print("Ones ", ones)
print("Zerohs ", zerohs)
print("Size of data ", data_len)

p_ones = float(ones)/float(data_len)
p_zerohs = float(zerohs)/float(data_len)

# print("P_Ones ", p_ones)
# print("P_Zerohs ", p_zerohs)

H = -p_zerohs*np.log2(p_zerohs) - (p_ones)*np.log2(p_ones)
print("\n")
print("The entropy of 1 bit is is ", H)
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

dict_2bit_p = {
"11":float(oneone)/float(data_len),
"10": float(onezero)/float(data_len),
"01": float(zeroone)/float(data_len),
"00": float(zerozero)/float(data_len)
}

H_2bit = 0
for key in dict_2bit_p:
    if dict_2bit_p[key] != 0:
        H_2bit -= dict_2bit_p[key]*np.log2(dict_2bit_p[key])

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
dict_3bit_p = {"0000": 0,
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

H_3bit = 0

for i in range(0, np.size(data_3bit)):
    key = data_3bit[i]
    if key in occurence_list_3bit:
        dict_3bit[key] += 1

for key in dict_3bit:
        dict_3bit_p[key] = dict_3bit[key]/np.size(data_3bit)

for key in dict_3bit:
    if dict_3bit_p[key] == 0:
        continue
    H_3bit -= dict_3bit_p[key]*np.log2(dict_3bit_p[key])

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

for key in dict_4bit:
        dict_4bit_p[key] = dict_4bit[key]/np.size(data_4bit)


for key in dict_4bit:
    if dict_4bit_p[key] == 0:
        continue
    H_4bit -= dict_4bit_p[key]*np.log2(dict_4bit_p[key])

print(f"\nThe entropy of four bits: {H_4bit}")


# 1f entropy rate
""" It is assumed that the random varables are iid."""

symbol_length = 127 #started on 15
n = symbols('n')
prob1 = [p_ones, p_zerohs]
prob2 = [dict_2bit_p[key] for key in dict_2bit_p]
prob3 = [dict_3bit_p[key] for key in dict_3bit_p]
prob4 = [dict_4bit_p[key] for key in dict_4bit_p]

hi1 = [-item*np.log2(item) for item in prob1]
hi2 = [-item*np.log2(item) for item in prob2 if item !=0]
hi3 = [-item*np.log2(item) for item in prob3 if item !=0]
hi4 = [-item*np.log2(item) for item in prob4 if item !=0]

f1 = 1/n*sum(hi1)
f2 = 1/n*sum(hi2)
f3 = 1/n*sum(hi3)
f4 = 1/n*sum(hi4)

entropy_rate1 = limit(f1,n,symbol_length)
entropy_rate2 = limit(f2,n,symbol_length)
entropy_rate3 = limit(f3,n,symbol_length)
entropy_rate4 = limit(f4,n,symbol_length)
print(f"\nThe entropy rate for 1 bit is: {entropy_rate1}")
print(f"\nThe entropy rate for 2 bit is: {entropy_rate2}")
print(f"\nThe entropy rate for 3 bit is: {entropy_rate3}")
print(f"\nThe entropy rate for 4 bit is: {entropy_rate4}")
print(f"A symbol length equal to 1 makes the entropy rate equal to the entropy.\
 Increasing the symbol lenght gives an entropy rate going towards zero. \
We have chosen a symbol length of {symbol_length}. ")
