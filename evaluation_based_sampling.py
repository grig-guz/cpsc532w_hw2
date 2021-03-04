from daphne import daphne
import primitives
from tests import is_tol, run_prob_test,load_truth
import torch
import torch.distributions as dist
import numpy as np
import time

def evaluate_program(ast, sig, env={}):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """

    if isinstance(ast, list) and isinstance(ast[0], list):
        for func in ast:
            if func[0] == 'defn':
                fname = func[1]
                fargs = func[2]
                fe = func[3]
                env[fname] = [fargs, fe]
        ast = ast[-1]

    if isinstance(ast, (int, float)):
        return torch.tensor(ast + 0.0), sig
    elif isinstance(ast, str) and ast in env:
        return env[ast], sig
    elif ast[0] in env:
        args, expr = env[ast[0]]
        env = {key:evaluate_program(val, sig, env)[0] for key, val in zip(args, ast[1:])}
        return evaluate_program(expr, sig, env)
    elif ast[0] == "+":
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.plus(arg1, arg2), sig
    elif ast[0] == 'sqrt':
        arg, sig = evaluate_program(ast[1], sig, env)
        return primitives.sqrt(arg),  sig
    elif ast[0] == '*':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.product(arg1, arg2),  sig
    elif ast[0] == '/':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.div(arg1, arg2),  sig
    elif ast[0] == 'vector':
        results = []
        for i in range(1, len(ast)):
            res, sig = evaluate_program(ast[i], sig)
            results.append(res)
        return primitives.vector(*results), sig
    elif ast[0] == 'get':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.get(arg1, arg2), sig
    elif ast[0] == 'put':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        arg3, sig = evaluate_program(ast[3], sig, env)
        return primitives.put(arg1, arg2, arg3), sig
    elif ast[0] == 'first':
        arg, sig = evaluate_program(ast[1], sig, env)
        return primitives.first(arg), sig
    elif ast[0] == 'second':
        arg, sig = evaluate_program(ast[1], sig, env)
        return primitives.second(arg), sig
    elif ast[0] == 'last':
        arg, sig = evaluate_program(ast[1], sig, env)
        return primitives.last(arg), sig
    elif ast[0] == 'append':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.append(arg1, arg2), sig
    elif ast[0] == 'cons':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.cons(arg1, arg2), sig
    elif ast[0] == 'conj':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2 = [evaluate_program(ast[i])[0] for i in range(2, len(ast))]
        return primitives.conj(arg1, arg2), sig
    elif ast[0] == 'hash-map':
        results = [evaluate_program(ast[i], sig, env)[0] for i in range(1, len(ast))]
        return primitives.hash_map(*results), sig
    elif ast[0] == 'normal':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.normal(arg1, arg2), sig
    elif ast[0] == 'beta':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.beta(arg1, arg2), sig
    elif ast[0] == 'dirac':
        arg1, sig = evaluate_program(ast[1], sig, env)
        return primitives.dirac(arg1), sig
    elif ast[0] == 'gamma':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.gamma(arg1, arg2), sig
    elif ast[0] == 'dirichlet':
        arg1, sig = evaluate_program(ast[1], sig, env)
        return primitives.dirichlet(arg1), sig
    elif ast[0] == 'exponential':
        arg1, sig = evaluate_program(ast[1], sig, env)
        return primitives.exponential(arg1), sig
    elif ast[0] == 'uniform':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.uniform(arg1, arg2), sig
    elif ast[0] == 'discrete':
        arg1, sig = evaluate_program(ast[1], sig, env)
        return primitives.discrete(arg1), sig
    elif ast[0] == 'flip':
        arg1, sig = evaluate_program(ast[1], sig, env)
        return primitives.discrete(torch.tensor([1 - arg1, arg1])), sig
    elif ast[0] == 'sample':
        arg1, sig = evaluate_program(ast[1], sig, env)
        return arg1.sample(), sig
    elif ast[0] == 'observe':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        sig += arg1.log_prob(arg2)
        return arg2, sig
    elif ast[0] == 'let':
        v1 = ast[1][0]
        e1_val, sig = evaluate_program(ast[1][1], sig, env)
        env[v1] = e1_val
        return evaluate_program(ast[2], sig, env)
    elif ast[0] == 'rest':
        arg1, sig = evaluate_program(ast[1], sig, env)
        return primitives.rest(arg1), sig
    elif ast[0] == 'mat-transpose':
        arg1, sig = evaluate_program(ast[1], sig, env)
        return primitives.mat_transpose(arg1), sig
    elif ast[0] == 'mat-tanh':
        arg1, sig = evaluate_program(ast[1], sig, env)
        return primitives.mat_tanh(arg1), sig
    elif ast[0] == 'mat-add':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.mat_add(arg1, arg2), sig
    elif ast[0] == 'mat-mul':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.mat_mul(arg1, arg2), sig
    elif ast[0] == 'mat-repmat':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        arg3, sig = evaluate_program(ast[3], sig, env)
        return primitives.mat_repmat(arg1, arg2, arg3), sig
    elif ast[0] == 'if':
        e1_val, sig = evaluate_program(ast[1], sig, env)
        if e1_val:
            return evaluate_program(ast[2], sig, env)
        else:
            return evaluate_program(ast[3], sig, env)
    elif ast[0] == 'and':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.and_f(arg1, arg2), sig
    elif ast[0] == 'or':
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        return primitives.or_f(arg1, arg2), sig
    elif ast[0] in ['<', '>', '<=', '>=', '=']:
        arg1, sig = evaluate_program(ast[1], sig, env)
        arg2, sig = evaluate_program(ast[2], sig, env)
        if ast[0] == '<':
            return arg1 < arg2, sig
        elif ast[0] == '>':
            return arg1 > arg2, sig
        elif ast[0] == '<=':
            return arg1 <= arg2, sig
        elif ast[0] == '>=':
            return arg1 >= arg2, sig
        elif ast[0] == '=':
            return arg1 == arg2, sig
    else:
        print(env)
        raise Exception(ast)

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast, 1)[0]



def run_deterministic_tests():

    for i in range(14,16):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        print(ast, truth)
        ret, sig = evaluate_program(ast)
        print(ret)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))

        print('Test passed')

    print('All deterministic tests passed')



def run_probabilistic_tests():

    num_samples=1e4
    max_p_value = 1e-4

    for i in range(1,7):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        print(ast)
        print(ast[0])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(ast)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)

    print('All probabilistic tests passed')

def likelihood_weighting(ast, num_samples):
    samples, weights = [], []
    for i in range(num_samples):
        val, sig =  evaluate_program(ast, 0)
        if len(val.shape) == 0:
            val = val.unsqueeze(dim=0)
        samples.append(val)
        weights.append(sig)
    samples = torch.stack(samples, dim=0).view(-1, samples[0].shape[0])
    weights = torch.tensor(weights).view(-1, 1)
    return samples.float(), weights

if __name__ == '__main__':

    #run_deterministic_tests()
    #run_probabilistic_tests()

    for i in range(5,6):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print(ast)
        now = time.time()
        samples, weights = likelihood_weighting(ast, 100000)
        print(time.time() - now)
        with open('is_' + str(i) + '.npy', 'wb') as f:
            np.save(f, samples.numpy())
        with open('is_weights_' + str(i) + '.npy', 'wb') as f:
            np.save(f, weights.numpy())

        print(samples.shape, weights.shape)
        print("Expectation: ")
        mean = ((samples * torch.exp(weights)) / torch.exp(weights).sum()).sum(dim=0)
        print(mean)
        print("Variance: ")
        variance = ((np.square(samples - mean) * torch.exp(weights)) / torch.exp(weights).sum()).sum(dim=0)
        print(variance)
