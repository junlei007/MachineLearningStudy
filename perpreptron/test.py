


for i in range(1,100000):
    s = []
    for j in range(1,i):
        if i%j==0:
            s.append(j)
    if sum(s)==i:
        print(i,s)