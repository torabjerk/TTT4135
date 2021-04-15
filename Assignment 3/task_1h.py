#Run length code

N = [7, 15, 31, 63, 127]

def printRLE(st):
    n = len(st)
    i = 0
    while i < n- 1:

        # Count occurrences of
        # current character
        count = 1
        while (i < n - 1 and
               st[i] == st[i + 1]):
            count += 1
            i += 1
        i += 1

        # Print character and its count
        print(st[i - 1] +
              str(count),
              end = "")


st = "wwwwaaadexxxxxxywww"
printRLE(st)
