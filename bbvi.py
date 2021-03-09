import torch
from utils import topological_sort

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
                self.proposals[node] = dist
                self.params += dist.Parameters()

        self.optimizer = torch.optim.Adam(self.params, lr=1e-2)
        self.step = 1

    def optimizer_step(self, grads):
        print(grads)
        print(self.params)
        count = 0
        for v in grads.keys():
            for i, param in enumerate(self.proposals[v].Parameters()):
                self.params[count].grad = -grads[v][i]
                count += 1
        self.optimizer.step()
        self.step += 1
        self.optimizer.zero_grad()

    def elbo_gradients(self, grads, logw):
        ghat = {}
        L = len(grads)
        for i in range(L):
            for v in self.proposals.keys():
                if v not in ghat:
                    ghat[v] = torch.tensor(grads[i][v]) * logw[i] / L
                else:
                    ghat[v] = torch.tensor(grads[i][v]) * logw[i] / L
        return ghat


    def run_bbvi(self, T, L):
        samples, all_weights = [], []
        for t in range(T):
            grads, weights = [], []
            print(self.params)
            for l in range(L):
                self.env['log_weight'] = 0
                self.grads = {}
                self.sample_from_joint()
                grads.append(self.grads)
                weights.append(self.env['log_weight'])
                samples.append(self.env['buffer'])
                self.optimizer.zero_grad()

            all_weights.extend(weights)
            elbo_grads = self.elbo_gradients(grads, weights)
            self.optimizer_step(elbo_grads)
        return samples, all_weights

    def bbvi_sample(self, v, prior_dist):
        dist = self.proposals[v]
        c = dist.sample()
        # TODO: Check if need munis sign!
        log_prob = dist.log_prob(c)
        log_prob.backward()
        self.grads[v] = [param.grad.clone().detach() for param in dist.Parameters()]
        self.env['log_weight'] += prior_dist.log_prob(c).detach() - log_prob.detach()
        return c

    def bbvi_observe(self, v, dist):
        val = self.graph['Y'][v]
        self.env['log_weight'] += dist.log_prob(val).detach()
        return val

    def sample_from_joint(self):
        "This function does ancestral sampling starting from the prior."
        self.env['buffer'] = {}
        for v in self.top_sort:
            dist = self.env['det_eval'](self.graph['P'][v][1], self.env)
            if v in self.graph['Y']:
                self.env['buffer'][v] = self.bbvi_observe(v, dist)
            else:
                self.env['buffer'][v] = self.bbvi_sample(v, dist)
