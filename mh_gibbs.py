import torch
from utils import *
import copy
import time
import numpy as np

class MHGibbs:
    def __init__(self, graph, env, num_samples):
        self.graph = graph[1]
        self.expr = graph[-1]
        self.env = env
        self.num_samples = num_samples
        self.top_sort = topological_sort(self.graph['V'], self.graph['A'])

    def find_Vx(self, x):
        Vx = [x]
        if x in self.graph['A']:
            Vx.extend(self.graph['A'][x])
        return Vx

    def accept(self, x, new_assign, old_assign):
        q = self.env['det_eval'](self.graph['P'][x][1], self.env)
        self.env['buffer'] = new_assign
        q_p = self.env['det_eval'](self.graph['P'][x][1], self.env)
        Vx = self.find_Vx(x)
        log_alpha = q_p.log_prob(old_assign[x]) - q.log_prob(new_assign[x])
        for v in Vx:
            self.env['buffer'] = new_assign
            log_alpha = log_alpha + self.env['det_eval'](self.graph['P'][v][1], self.env).log_prob(torch.tensor(self.env['buffer'][v]))
            self.env['buffer'] = old_assign
            log_alpha = log_alpha - self.env['det_eval'](self.graph['P'][v][1], self.env).log_prob(torch.tensor(self.env['buffer'][v]))
        return torch.exp(log_alpha)

    def gibbs_step(self):
        for x in set(self.graph['V']).difference(set(self.graph['Y'])):
            new_x = self.env['det_eval'](self.graph['P'][x], self.env)
            X_p = copy.deepcopy(self.env['buffer'])
            X_p[x] = new_x
            alpha = self.accept(x, X_p, self.env['buffer'])
            u = torch.rand(1)
            if u < alpha:
                self.env['buffer'] = X_p

        return self.env['buffer']

    def gibbs(self):
        # Assign initial values to V
        sample_from_prior(self.graph, self.env)
        samples = []
        for i in range(self.num_samples):
            sample = self.gibbs_step()
            samples.append(sample)
        return samples

    def joint_logprob(self, X):
        "Evaluate joint density"
        log_prob = 0
        inputs = []
        self.env['buffer'] = X
        for node in self.top_sort:
            dist = self.env['det_eval'](self.graph['P'][node][1], env=self.env)
            val = torch.tensor(X[node])
            log_prob += dist.log_prob(val)
        return log_prob.detach().numpy()

    def run_gibbs(self, prog_num):
        now = time.time()
        samples = self.gibbs()
        print(time.time() - now)
        results = []
        logprobs = []
        for sample in samples:
            self.env['buffer'] = sample
            result = self.env['det_eval'](self.expr, self.env)
            logprob = self.joint_logprob(sample)
            logprobs.append(logprob)
            if len(result.shape) == 0:
                result = result.unsqueeze(dim=0)
            results.append(result)
        results = results[5000:]
        logprobs = logprobs[5000:]
        results = torch.stack(results, dim=0).view(-1, results[0].shape[0]).float()
        with open('mh_' + str(prog_num) + '.npy', 'wb') as f:
            np.save(f, np.array(results))
        with open('mh_logprobs_' + str(prog_num) + '.npy', 'wb') as f:
            np.save(f, np.array(logprobs))

        print("Expectation: ")
        mean = torch.mean(results, dim=0)
        print(mean)
        print("Variance: ")
        variance = torch.var(results, dim=0)
        print(variance)
