#Run length code
import numpy as np
#data = np.loadtxt("data1.txt")
#data2 = np.loadtxt("data2.txt")
#data3 = np.loadtxt("data3.txt")
data = "1110010100010000"
N = [7, 15, 31, 63, 127]
len_data = len(data)
C = ""
print(len_data)

if data[0] != 0:
    C += "0"

i = 0
while i < len_data-1:
    count = 1
    while (i < len_data-1 and data[i] == data[i+1]):
        #print("hello")
        count += 1
        i+=1
    i+= 1
    
    C += str(count)


print(C)
