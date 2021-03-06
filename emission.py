import copy

class emission(object):
    def __init__(self):
        self.X = []
        self.Y = []
        self.testX = []
        self.tokenizedX = []
        self.optY = []
        self.word_count = {}
        self.prob = {} # a dict of emission probablities: {label1: {word1: 0.3, word2: 0.2}}

    def process(self, file):
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
        self.X = X
        self.Y = Y

    def process_input(self, infile):
        with open(infile) as f:
            data = f.read().splitlines()
            x = []

        for i in data[:300]:
            if i != '':
                x.append(i)
            else:
                self.testX.append(x)
                x = []

    def tokenize(self, X):
        print(X)
        k = 3
        token = '#UNK#'
        word_count = {}
        #replace with UNK
        for i in range(len(X)):
            for j in range(len(X[i])):
                word = X[i][j]
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
        print(word_count)
        for i in range(len(X)):
            for j in range(len(X[i])):
                word = X[i][j]
                if word in word_count.keys():
                    if word_count[word] < k:
                        X[i][j] = token
                        del word_count[word]
        self.word_count = word_count
        self.X = X
        return X

    def tokenize_test(self, X):
        clone = copy.deepcopy(X)
        for i in range(len(X)):
            for j in range(len(X[i])):
                foundFlag = False
                if clone[i][j] in self.word_count.keys():
                    foundFlag = True
                    continue
                if not foundFlag:
                    clone[i][j] = "#UNK#"
        print(X)
        return clone


    def get_emission_prob(self, X, Y):
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

        self.prob = result

    def get_opt_tags(self, X, em):
        result = []
        for i in range(len(X)):
            result.append([])
            for j in range(len(X[i])):
                temp = {}
                x = X[i][j]
                for label, obs in em.items():
                    if x in obs.keys():
                        temp[label] = obs[x]

                result[i].append(max(temp, key=temp.get))
        self.optY = result

    def score(self):


    def print_out(self, train, infile, outfile):
        self.process(train)
        self.process_input(infile)
        self.tokenize(self.X)
        self.tokenizedX = self.tokenize_test(self.testX)
        self.get_emission_prob(self.X, self.Y)
        self.get_opt_tags(self.tokenizedX, self.prob)

        outfile = open(outfile, 'w')
        for i in range(len(self.testX)):
            for j in range(len(self.testX[i])):
                obs = self.testX[i][j]
                if j == len(self.testX[i])-1:
                    outfile.write("\n")
                else:
                    outfile.write(obs + " " +  self.optY[i][j] + "\n")
