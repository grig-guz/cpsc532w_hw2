import torch
import copy
from utils import topological_sort, sample_from_prior
import primitives
import numpy as np
import time

class HMC:

    def __init__(self, graph, env):
        self.graph = graph[1]
        self.expr = graph[-1]
        self.env = env
        self.top_sort = topological_sort(self.graph['V'], self.graph['A'])
        self.vars_to_sample = [var for var in self.top_sort if var not in self.graph['Y']]
        self.num_to_sample = len(self.vars_to_sample)

    def joint_logprob(self, X):
        "Evaluate joint density"
        log_prob = 0
        inputs = []
        old_buffer =  self.env['buffer']
        self.env['buffer'] = X
        for node in self.top_sort:
            dist = self.env['det_eval'](self.graph['P'][node][1], env=self.env)
            if node.startswith("sample"):
                val = X[node].clone().detach().requires_grad_(True)
                inputs.append(val)
            else:
                val = torch.tensor(X[node])
            log_prob += dist.log_prob(val)
        self.env['buffer'] = old_buffer
        return log_prob, inputs

    def nabla_u(self, X):
        U, inputs = self.joint_logprob(X)
        U.backward()
        grads = torch.tensor([val.grad for val in inputs])
        return grads

    def hamiltonian(self, X, R, M):
        K = 0.5*torch.sum(R * R / M)
        U, _ = self.joint_logprob(X)
        return K - U

    def leapfrog(self, x_t, r_0, T, eps):
        r_h = r_0 - 0.5 * eps * self.nabla_u(x_t)
        for t in range(T - 1):
            for i, var in enumerate(self.vars_to_sample):
                x_t[var] += eps * r_h[i]
            r_h = r_h - eps * self.nabla_u(x_t)
        for i, var in enumerate(self.vars_to_sample):
            x_t[var] += eps * r_h[i]
        r_h -= - 0.5 * eps * self.nabla_u(x_t)
        return x_t, r_h

    def run_hmc(self, S, T, eps, M, prog_num):
        now = time.time()
        samples = []
        x = self.env['buffer']
        sample_from_prior(self.graph, self.env)
        mean = torch.zeros(self.num_to_sample)
        M = torch.ones(self.num_to_sample) * M

        for s in range(S):
            r_s = primitives.normal(mean, M).sample()
            x_p, r_p = self.leapfrog(copy.deepcopy(self.env['buffer']), r_s, T, eps)
            u = torch.rand(1)

            if u < torch.exp(-self.hamiltonian(x_p, r_p, M) + self.hamiltonian(self.env['buffer'], r_s, M)):
                self.env['buffer'] = x_p
                samples.append(x_p)
            else:
                samples.append(self.env['buffer'])
        results = []
        logprobs = []
        print(time.time() - now)

        for sample in samples:
            self.env['buffer'] = sample
            result = self.env['det_eval'](self.expr, self.env)
            logprob = self.joint_logprob(sample)[0].detach().numpy()
            logprobs.append(logprob)
            if len(result.shape) == 0:
                result = result.unsqueeze(dim=0)
            results.append(result)

        results = torch.stack(results, dim=0).view(-1, results[0].shape[0]).float()
        with open('hmc_' + str(prog_num) + '.npy', 'wb') as f:
            np.save(f, np.array(results))
        with open('hmc_logprobs_' + str(prog_num) + '.npy', 'wb') as f:
            np.save(f, np.array(logprobs))

        print("Expectation: ")
        mean = torch.mean(results, dim=0)
        print(mean)
        print("Variance: ")
        variance = torch.var(results, dim=0)
        print(variance)

        return samples
