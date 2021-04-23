import numpy as np
from sympy import symbols, limit
from sympy.solvers import solve

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
"11":float(oneone)/float(data_len/2),
"10": float(onezero)/float(data_len/2),
"01": float(zeroone)/float(data_len/2),
"00": float(zerozero)/float(data_len/2)
}

""" Markov Modelling"""
P = np.array([[dict_2bit_p["00"]/p_zerohs, dict_2bit_p["01"]/p_zerohs], [dict_2bit_p["10"]/p_ones,dict_2bit_p["11"]/p_ones]])

print(f"Transition matrix: {P}")

#one_step_transition = np.array([[7/8, 1/8], [1/8, 7/8]])
p1 = symbols('p1')
f_p1 = p1*P[0][0]+(1-p1)*P[0][1]-p1
q_p1 = solve(f_p1,p1)
q_p1 = q_p1[0]
q_p2 = 1 - q_p1
q = [q_p1,2, q_p2,2]
print(f"Steady state vector: {q}")

q_p1 = float(q_p1[0])
print(f"q_p1: {q_p1}")
H_markov = -q_p1*np.log2(q_p1) - (q_p2)*np.log2(q_p2)

"""
def steady_state_prop(p):
    dim = p.shape[0]
    q = (p-np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    print(f"q: {q}")
    QTQ = np.dot(q, q.T)
    print(f"QTQ: {QTQ}")
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ,bQT)

steady_state_matrix = steady_state_prop(P.transpose())

print (steady_state_matrix)
"""
