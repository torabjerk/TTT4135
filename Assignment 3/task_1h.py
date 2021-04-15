#Run length code
import numpy as np
#data = np.loadtxt("data1.txt")
#data2 = np.loadtxt("data2.txt")
#data3 = np.loadtxt("data3.txt")

data = "11100101000000000"
N = [7, 15, 31, 63, 127]

def run_lenght(data, n):
    len_data = len(data)
    C = ""

    if data[0] != 0:
        C += "0"

    i = 0
    while i < len_data-1:
        count = 1
        while (i < len_data-1 and data[i] == data[i+1]):
            #print("hello")
            count += 1
            i+=1

        if count > n:
            M = count - n
            count_str = f"{n}{data[i]}{M}"
            print(count_str)
        else:
            count_str = str(count)

        C += count_str
        i+= 1

    return C


code = run_lenght(data, N[0])
print(code)
