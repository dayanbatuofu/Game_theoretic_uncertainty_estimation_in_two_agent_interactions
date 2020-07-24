import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.rcParams['text.usetex'] = True


inf_df = pd.read_pickle('./start_slice/neutral_ne_a1.pkl')
g2_df = pd.read_pickle('./start_slice/g2_ne_a1.pkl')

inf_states = list(inf_df['start_states'])

inf_p2_r = list(inf_df['p2_reward'])
g2_p2_r = list(g2_df['p2_reward'])
difference = []

# zip_object = zip(inf_p2_r, g2_p2_r)
zip_object = zip(g2_p2_r, inf_p2_r)
for list1_i, list2_i in zip_object:
    difference.append(list1_i - list2_i)

max_diff = max(difference)
min_diff = min(difference)
min_idx = difference.index(min_diff)
print(min_idx)
print(inf_p2_r[min_idx], g2_p2_r[min_idx], inf_states[min_idx])

print(max_diff, min_diff)

ctr = 0
for val in difference:
    if val < 0:
        ctr +=1
print(ctr)

temp = []

for i in range(1001):
    temp.append(1 * i)

# s1 = list(reversed(temp))
# print(s1)
s2 = []

plt.hist(difference, bins=temp)
plt.axis([0, 30, 0, 300])
plt.title('Histogram for Sparsity')
plt.xlabel('Loss between Non-Graceful and Graceful')
plt.ylabel('count')

plt.show()