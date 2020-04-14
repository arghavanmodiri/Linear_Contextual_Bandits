import multiprocessing as mp

def ss(a,b,c):
    return [a+b+c, a*b*c]

if __name__ == '__main__':
    print(ss(2,1,1))
    print(ss(3,2,1))
    print(ss(4,2,1))
    a,b=ss(4,2,1)
    print(a," ",b)

    pool = mp.Pool()
    tasks = [*zip(range(1,5), [3]*4, [2]*4)]
    results = pool.starmap(ss, tasks)
    print(results) #[[6,6], [7,12], [8, 18], [9, 24]]

    aval=[el[0] for el in results]
    dovom=[el[1] for el in results]
    print(aval) #[6, 7, 8, 9]
    print(dovom) #[6, 12, 18, 24]


