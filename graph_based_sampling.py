import torch
import torch.distributions as dist
import copy
from daphne import daphne
import numpy as np
import primitives
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {'normal': dist.Normal,
        'beta': primitives.beta,
        'exponential': primitives.exponential,
        'uniform': primitives.uniform,
       'sqrt': torch.sqrt,
       'vector': primitives.vector,
       'put': primitives.put,
       'hash-map': primitives.hash_map,
       'sample*': lambda x: x.sample(),
       'if': lambda x, y, z: y if x else z,
       '<': lambda x, y: x < y,
       '>': lambda x, y: x > y,
       'observe*': lambda x, y: None,
       '+': primitives.plus,
       '*': primitives.product,
       'discrete':primitives.discrete,
       'get':primitives.get,
       'mat-transpose': primitives.mat_transpose,
       'mat-tanh': primitives.mat_tanh,
       'mat-add': primitives.mat_add,
       'mat-mul': primitives.mat_mul,
       'mat-repmat': primitives.mat_repmat
}


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        if op == 'hash-map':
            new_args = []
            for arg in args:
                new_args.extend(arg)
            args = new_args
        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    elif exp in env:
        return env[exp]
    else:
        print("here", exp)
        raise("Expression type unknown.", exp)

def topological_sort_helper(node, visited, stack, nodes, edges):
    visited.add(node)
    if node in edges:
        for child in edges[node]:
            if child not in visited:
                topological_sort_helper(child, visited, stack, nodes, edges)
    stack.append(node)

def topological_sort(nodes, edges):
    visited = set()
    stack = []
    for node in nodes:
        if node not in visited:
            topological_sort_helper(node, visited, stack, nodes, edges)
    return stack[::-1]

def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    expr = graph[-1]
    graph = graph[1]
    top_sort = topological_sort(graph['V'], graph['A'])
    for node in top_sort:
        env[node] = deterministic_eval(graph['P'][node])
    return deterministic_eval(expr)


def get_stream(graph):
    """Return a stream of prior samples
    Args:
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)

#Testing:

def run_deterministic_tests():

    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        print(graph)
        ret = deterministic_eval(graph[-1])
        print(ret)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))

        print('Test passed')

    print('All deterministic tests passed')



def run_probabilistic_tests():

    #TODO:
    num_samples=1e4
    max_p_value = 1e-4

    for i in range(1,7):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        print(graph)
        stream = get_stream(graph)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)

    print('All probabilistic tests passed')




if __name__ == '__main__':


    #run_deterministic_tests()
    #run_probabilistic_tests()

    for i in range(1,5):
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        acc = []
        for _ in range(1000):
            acc.append(sample_from_joint(graph))
        #
        if i == 4:
            with open(str(i) + "_graph.npy", 'wb') as f:
                for j in range(4):
                    part_acc = []
                    for k in range(1000):
                        part_acc.append(acc[k][j].numpy())
                    print(part_acc)
                    np.save(f, np.stack(part_acc))
        else:
            acc = torch.stack(acc)
            with open(str(i) + "_graph.npy", 'wb') as f:
                print(acc)
                np.save(f, acc.numpy())
