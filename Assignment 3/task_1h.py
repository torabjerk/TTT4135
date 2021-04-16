#Run length code
import numpy as np
data = np.loadtxt("data3.txt")
#data2 = np.loadtxt("data2.txt")
#data3 = np.loadtxt("data3.txt")

#data = "11100101000000000"
N = [7, 15, 31, 63, 127]

def run_lenght_n(data, n):
    len_data = len(data)
    C = ""

    if data[0] != 0:
        C += "0"
    i = 0
    while i < len_data-1:
        count = 1
        while (i < len_data-1 and data[i] == data[i+1]):
            count += 1
            i+=1

        if count > n:
            M = count - n
            count_str = f"{n}{data[i]}{M}"
        else:
            count_str = str(count)

        C += count_str
        i+= 1

    return C


code = []
for item in N:
    code.append(run_lenght_n(data,item))

coding_gain = {}
for i in range(len(N)):
    coding_gain[N[i]] = len(data)/len(code[i])

print("Coding gain:",coding_gain)

#print(code[0])
#print(len(code))
## # TODO: Calculate coding gain
