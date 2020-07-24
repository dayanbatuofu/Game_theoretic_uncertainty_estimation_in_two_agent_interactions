import random, math

eps_start = 1
eps_end = 0.01
eps_decay = 60000
random_ct = 0
eta = 0.6
max_frames = 5000000

prev_eps = 0
for i in range(1, max_frames+1):
  if random.random() <= eta:
    epsilon = float(eps_end+(eps_start-eps_end)*math.exp(-1.* (i/eps_decay)))
    if random.random() <= epsilon:
      random_ct += 1
      if epsilon == eps_end:
        print('frame: {}, eps: {}, prev_eps: {}'.format(i, epsilon, prev_eps))
      # break
    prev_eps = epsilon

print(random_ct, max_frames-random_ct)
