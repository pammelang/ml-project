class worditem(object):
    def __init__(self, word, prob, m):
        self.word = word
        self.score = prob
        self.path = m
class NBest(object):
    def __init__(self, n):
        self.elements=[None,]
        self.N = n
# add new tag, its probability and number into the array
# maintain this entire array with a heap
    def add(self, word, prob, m):
        a = worditem(word,prob,m)
        self.elements.append(a)
        i = len(self.elements)-1
        while self.elements[i/2] is not None and self.elements[i/2].score<a.score:
            self.elements[i] = self.elements[i/2]
            i = i/2
        self.elements[i] = a
    def deleteMax(self) :
        a = self.elements[1]
        b = self.elements.pop()
        i = 1;
        while i*2<len(self.elements):
            child = i*2
            if child+1<len(self.elements) and self.elements[child+1].score>self.elements[child].score:
                child+=1
            if self.elements[child].score > b.score:
                self.elements[i] = self.elements[child]
            else:
                break;
            i = child
# avoid indexing zero
        if i<len(self.elements):
            self.elements[i] = b
        return a
# get the first N from them
    def best(self):
        belement = []
        i = 1
        while i<=self.N and len(self.elements)>1:
            belement.append(self.deleteMax())
            i+=1
        self.elements = belement
# get the first i-th label in the array
    def pop(self,i):
        return self.elements.pop(i)

class transition(object):
    def __init__(self):
        self.X = []
        self.Y = []
        self.testX = []
        self.tokenizedX = []
        self.optY = []
        self.prob = {}

    def train(file):
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

        self.prob = result
        return result

    def viterbi(self, e, t, infile, outfile, p=True):
        # Y = [ <x1> {'tag1': prob1,}, <x2> {'tag2': prob, 'tag3': prob},]
        # path = [tag1, tag2,] based on input
        try:
            inFile = open(infile, 'r')
            outFile = open(outfile, 'w')
            lines = inFile.readlines()

            START = True
            Y, path = [], []

            rng = range(len(lines))
            for i in rng:
                line = lines[i]
                tags = {}
                x = line.strip()
                if START:
                    for tag, obs in e.items():
                        for combi in t:
                            if combi[0] == 'START' and combi[1] == tag:
                                tags[tag] = combi[2]*1.0 * obs[x]
                    Y.append(tags)
                    path.append['']
                    START = False
                elif line == '\n':
                    maxProb = 0
                    prevTags = ""
                    for tag, obs in e.items():
                        for combi in t:
                            if combi[0] == tag and combi[1] == 'STOP':
                                tags[tag] = combi[2] * Y[i - 1][tag]
                    if tags[tag] >= maxProb:
                        maxProb = tags[tag]
                        prevTags = tag
                    Y.append(tags)
                    path.append(tag)
                    START = True
                else:
                    prevTagss = {}
                    for tag, obs in e.items():
                        for tagg, obss in e.items():
                            for combi in t:
                                if combi[0] == tagg and combi[1] == tag:
                                    prob = Y[i - 1][tagg] * combi[2] * obs[x]
                                    # print ftag,"-->",tag,":",prob
                            if (tag not in tags) or (tags[tag] <= prob) :
                                tags[tag] = prob
                                prevTags[tag] = tagg
                    Y.append(tags)
                    path.append(prevTags)
            rng.reverse()
            resultPath = []
            lastTag = ""
            for i in rng:
                if lines[i] == '\n':
                    if isinstance(path[i], str):
                        lastTag = path[i]
                        resultPath.append(lastTag)
                    else:
                        raise RuntimeError("Last tag is not defined")
                elif not path[i]:
                    x = lines[i].strip()
                    if isinstance(path[i], dict):
                        lastTag = path[i][lastTag]
                        resultPath.append(lastTag)
            output = []
            for line in lines:
                if line == '\n':
                    output.append('\n')
                else:
                    x = line.strip()
                    tag = resultPath.pop()
                    output.append(x + ' ' + tag + '\n')
            outFile.writelines(output)
        except(IOError, e):
            print(e)
            exit(0)
        finally:
            if inFile:
                inFile.close()
            if outFile:
                outFile.close()



    #
    # def best_viterbi(self, e, t, infile, outfile, n=5, p=True):
    #     try:
    #         inFile = open(infile, 'r')
    #         outFile = open(outfile, 'w')
    #
    #         lines = inFile.readlines()
    #         count = range(len(lines))
    #         X, Y = [], [] # Y
    #         START = True
    #
    #         for i in count:
    #             line = inData[i]
    #             # computes all possible tags and their
    #             # probabilities given their previous tag paths
    #             if START:
    #                 tags = {}
    #                 x = line.strip
    #                 for tag, obs in e.items():
    #                     for combi in t:
    #                         if combi[0] == 'START' and combi[1] == tag:
    #                             tags[tag] = combi[2]*1.0 * e[tag][x]
    #                 Y.append(tags)
    #                 START = False
    #             elif line == '\n':
    #                 START = True
    #                 nb = NBest(best) # initializes the best-sort heap
    #                 for tag, obs in e.items():
    #                     if isinstance(Y[i - 1][tag], float):
    #                         raise RuntimeError("Sentence has no words")
    #                     else:
    #                         b = Y[i - 1][tag]
        #
        #
        #
        #
        # except IOError, error:
        #     print error
        #     exit(0)
        # finally:
        #     if inFile:
        #         inf.close()
        #     if outFile:
        #         outf.close()
