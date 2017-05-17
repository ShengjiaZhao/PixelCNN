from matplotlib import pyplot as plt

reader = open('stein_logdet_log', 'r')

dim = []
sample = []
logdet = []
while True:
    line = reader.readline().split()
    if len(line) < 3:
        break
    dim.append(float(line[0]))
    sample.append(float(line[1]))
    logdet.append(float(line[2]))


# arr = [(d, l) for d, l, s in zip(dim, logdet, sample) if s == 1280]
# d_list = [item[0] for item in arr]
# l_list = [item[1] / item[0] for item in arr]
# plt.plot(d_list, l_list)
# plt.show()

arr = [(s, l) for d, l, s in zip(dim, logdet, sample) if d == 8]
s_list = [item[0] for item in arr]
l_list = [item[1] for item in arr]
plt.plot(s_list, l_list)
plt.show()