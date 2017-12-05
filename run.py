import emission as em
import transition as t

def main():
    e = em.emission()
    e.print_out('train', 'dev.in', 'dev.p2.out')

    t = t.transition()
    t.train('train')
    print(t.get_trans_params(t.Y))

def process(file):
    with open(file) as f:
        data = f.read().splitlines()
    X, Y, x, y = [], [], [], []

    for i in data[:300]:
        if i != '':
            o = i.split(' ', 2)
            x.append(o[0])
            y.append(o[1])
        else:
            X.append(x)
            Y.append(y)
            x, y = [], []
    return(X ,Y)

def print_out(X, em):
    outfile = open('dev.p2.out', 'w')
    opt_tags = get_opt_tag(X, em)
    for i in range(len(X)):
        for j in range(len(X[i])):
            obs = X[i][j]
            if j == len(X[i])-1:
                outfile.write("\n")
            else:
                outfile.write(obs + " " +  opt_tags[i][j] + "\n")

def get_trans_params(Y):
    count, permutations, result = [['START', 'STOP'],[0,0]], [], []
    # permutations = a set of tuples (i, j, count) - eg [['START', 'O', 32], ..]
    # count = (labels, count) - eg [['START', 'A'], [20, 12]]
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            label = Y[i][j]

            ### to increment count
            if j == 0: # START
                count[1][0] += 1
            elif j == len(Y[i])-1 :  # STOP
                count[1][1] += 1
            if label not in count[0]: # if label not yet in labels
                count[0].append(label)
                count[1].append(1)
            else:
                index = count[0].index(label)
                count[1][index] += 1

            # to increment permutation
            foundFlag = False
            if not permutations:
                permutations.append(['START', Y[i][j], 1])
                foundFlag = True
                continue
            for p in permutations:
                if j != 0 and Y[i][j-1] == p[0] and Y[i][j] == p[1]:
                    p[2] += 1 # if the permutation is found
                    foundFlag = True
                    break
                elif j == 0 and p[0] == 'START' and Y[i][j] == p[1]:
                    p[2] += 1
                    foundFlag = True
                    break
                elif j == len(Y[i]) - 1 and Y[i][j] == p[0] and p[1] == 'STOP':
                    p[2] +=1
                    foundFlag = True
                    break
                else: continue
            if foundFlag == False:
                if j == 0:
                    new_p = ['START', Y[i][j], 1]
                    permutations.append(new_p)
                elif j == len(Y[i]) - 1:
                    new_p = [Y[i][j], 'STOP', 1]
                    permutations.append(new_p)
                else:
                    new_p = [Y[i][j-1], Y[i][j], 1]
                    permutations.append(new_p)

    for i in range(len(Y)):
        result.append([])
        for j in range(len(Y[i])):
            for p in permutations:
                if j == 0 and p[0] == 'START' and p[1] == Y[i][j]:
                    result[i].append(p[2]/count[1][0])

                if p[0]== Y[i][j-1] and p[1] == Y[i][j]:
                    result[i].append(p[2]/count[1][count[0].index(p[0])])

                if j == len(Y[i])-1 and p[0] == Y[i][j] and p[1] == 'STOP':
                    result[i].append(p[2]/count[1][1])

    return result

def tokenize_obs(X):
    k = 3
    token = '#UNK#'
    word_count = {}
    newX = X[:]
    #replace with UNK
    for i in range(len(X)):
        for j in range(len(X[i])):
            word = X[i][j]
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1

    for i in range(len(X)):
        for j in range(len(X[i])):
            word = X[i][j]
            if word_count[word] < k:
                newX[i][j] = token
    return newX


def get_emission_params(X,Y):
    count, permutations, result = [], {}, []
    # count = list of labels & count eg. [['A', 'B'], [3, 1]]
    # permutation = dict of labels & obs eg. {'label': {'obs1': 3, 'obs2': 1}}
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            obs = X[i][j]
            label = Y[i][j]

            ### to init count
            if not count:
                count = [[label], [1]]
            elif label in count[0]:
                count[1][count[0].index(label)] += 1
            else:
                count[0].append(label)
                count[1].append(1)

            ### to init permutation
            if not permutations or label not in permutations:
                permutations[label] = {obs: 1}
            elif label in permutations:
                if obs in permutations[label]:
                    permutations[label][obs] += 1
                else:
                    permutations[label][obs] = 1

    result = {}
    for label in permutations:
        result[label] = {}
        for obs in permutations[label]:
            result[label][obs] = permutations[label][obs]/count[1][count[0].index(label)]

    return result

def get_opt_tag(X, emParams):
    result = []
    for i in range(len(X)):
        result.append([])
        for j in range(len(X[i])):
            temp = {}
            obs = X[i][j]
            for label in emParams:
                if obs in emParams[label]:
                    temp[label] = emParams[label][obs]

            result[i].append(max(temp, key=temp.get))
    return result

def viterbi(obs):
    n = len(obs)
    trellis = np.zeros((self.N, len(obs)))
    backpt = np.ones((self.N, len(obs)), 'int32') * -1
    trellis[:, 0] = np.squeeze(self.initialProb * self.Obs(obs[0]))
    for t in xrange(1, len(obs)):
        trellis[:, t] = (trellis[:, t-1, None].dot(self.Obs(obs[t]).T) * self.transProb).max(0)
        backpt[:, t] = (np.tile(trellis[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)
    tokens = [trellis[:, -1].argmax()]
    for i in xrange(len(obs)-1, 0, -1):
        tokens.append(backpt[tokens[-1], i])
    return tokens[::-1]

# print(get_opt_tags(X[:100], e))
