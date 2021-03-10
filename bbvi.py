import torch
from utils import topological_sort
import numpy as np
import time

class BBVI:

    def __init__(self, graph, env):
        self.env = env
        self.graph = graph[1]
        self.expr = graph[-1]
        self.top_sort = topological_sort(self.graph['V'], self.graph['A'])
        self.proposals = {}
        # Initialize proposal distributions
        self.params = []
        for node in self.top_sort:
            if node not in self.graph['Y']:
                dist = self.env['det_eval'](self.graph['P'][node][1], self.env).make_copy_with_grads()
                self.env['buffer'][node] = dist.sample().detach()
                self.proposals[node] = dist
                self.params += dist.Parameters()

        self.optimizer = torch.optim.Adam(self.params, lr=1e-2)

    def optimizer_step(self, grads):
        count = 0
        #print("Grads:", grads)
        #print("Params", self.params)
        #print(self.top_sort)
        for v in self.top_sort:
            if v not in self.graph['Y']:
                if self.params[count].dim() > 0:
                    self.params[count].grad = -grads[v]
                    count += 1
                else:
                    for i, param in enumerate(self.proposals[v].Parameters()):
                        self.params[count].grad = -grads[v][i]
                        count += 1
        self.optimizer.step()
        self.optimizer.zero_grad()

    def elbo_gradients(self, grads, logw):
        ghat = {}
        L = len(grads)
        for v in self.proposals.keys():
            for i in range(L):
                if len(grads[i][v]) == 1:
                    grad = grads[i][v][0]
                else:
                    grad = torch.tensor(grads[i][v])

                if v not in ghat:
                    ghat[v] = grad * logw[i] / L
                else:
                    ghat[v] += grad * logw[i] / L
        return ghat

    def bbvi_sample(self, v, prior_dist, do_is):
        dist = self.proposals[v]
        c = dist.sample()
        if do_is:
            return c
        log_prob = dist.log_prob(c)
        log_prob.backward()
        self.grads[v] = [param.grad.clone().detach() for param in dist.Parameters()]
        self.env['log_weight'] += prior_dist.log_prob(c).detach() - log_prob.detach()
        return c

    def bbvi_observe(self, v, dist):
        val = self.graph['Y'][v]
        self.env['log_weight'] += dist.log_prob(val).detach()
        return val

    def sample_from_joint(self, do_is=False):
        "This function does ancestral sampling starting from the prior."
        self.env['buffer'] = {}
        for v in self.top_sort:
            dist = self.env['det_eval'](self.graph['P'][v][1], self.env)
            if v in self.graph['Y']:
                self.env['buffer'][v] = self.bbvi_observe(v, dist)
            else:
                self.env['buffer'][v] = self.bbvi_sample(v, dist, do_is)

    def run_bbvi(self, T, L, prog_num):
        all_elbos = []
        now = time.time()
        for t in range(T):
            grads, elbos = [], []
            if t % 500 == 0:
                print(t)
            for l in range(L):
                self.env['log_weight'] = 0
                self.grads = {}
                self.sample_from_joint()
                grads.append(self.grads)
                elbos.append(self.env['log_weight'])
                self.optimizer.zero_grad()

            elbo = sum(elbos) / L
            all_elbos.append(elbo)
            elbo_grads = self.elbo_gradients(grads, elbos)
            self.optimizer_step(elbo_grads)
        with open('bbvi_' + str(prog_num) + '.npy', 'wb') as f:
            np.save(f, np.array(all_elbos))
        print(time.time() - now)
        results, weights = [[], [], [], []], []
        for i in range(100):
            self.env['log_weight'] = 0
            self.env['buffer'] = {}
            self.sample_from_joint(do_is=True)
            result = self.env['det_eval'](self.expr, self.env)

            for i in range(4):
                results[i].append(np.array(result[i]))

            """
            result = torch.tensor(result)
            if result.dim() == 0:
                result = result.unsqueeze(0)
            results.append(result)
            """
            weights.append(torch.exp(self.env['log_weight']))
        for i in range(4):
            with open('bbvi_' + str(prog_num) + str(i) + 'results.npy', 'wb') as f:
                np.save(f, np.array(results[i]))
        with open('bbvi_' + str(prog_num) + 'weights.npy', 'wb') as f:
            np.save(f, np.array(weights))

        results = torch.stack(results, dim=0).view(-1, results[0].shape[0]).float()
        weights = torch.stack(weights).unsqueeze(1) / sum(weights)
        print("Expectation: ")
        mean = torch.sum(results * weights, dim=0)
        print(mean)
        print(self.params)
        return results, weights
