import emission as em
import transition as t

e = em.emission()
# e.process(train)
# e.process_input(infile)
# e.tokenize(self.X)
# e.tokenizedX = self.tokenize_test(self.testX)
# e.get_emission_prob(self.X, self.Y)
e.print_out('train', 'dev.in', 'dev.p2.out')

t = t.transition()
t.train('train')
print(t.get_trans_params(t.Y))
