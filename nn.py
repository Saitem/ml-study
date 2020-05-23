from numpy import exp, array, random, dot

train_inp = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
train_out = array([[0, 1, 1, 0]]).T

random.seed(1)

syn_weights = 2 * random.random((3, 1)) - 1

for interation in range(10000):
    output = 1 / (1 + exp(-(dot(train_inp, syn_weights))))
    syn_weights += dot(train_inp.T, (train_out - output) * output * (1 - output))

print(1 / (1 + exp(-(dot(array([1, 0, 0]), syn_weights)))))
# print(syn_weights)