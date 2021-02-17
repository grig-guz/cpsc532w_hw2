from daphne import daphne
import primitives
from tests import is_tol, run_prob_test,load_truth
import torch
import torch.distributions as dist
import numpy as np
# TODO NTH

def evaluate_program(ast, env={}):
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
        return torch.tensor(ast + 0.0), None
    elif isinstance(ast, str) and ast in env:
        return env[ast], None
    elif ast[0] in env:
        args, expr = env[ast[0]]
        env = {key:evaluate_program(val, env)[0] for key, val in zip(args, ast[1:])}
        return evaluate_program(expr, env)
    elif ast[0] == "+":
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.plus(arg1, arg2), None
    elif ast[0] == 'sqrt':
        arg, _ = evaluate_program(ast[1], env)
        return primitives.sqrt(arg),  None
    elif ast[0] == '*':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.product(arg1, arg2),  None
    elif ast[0] == '/':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.div(arg1, arg2),  None
    elif ast[0] == 'vector':
        results = [evaluate_program(ast[i])[0] for i in range(1, len(ast))]
        return primitives.vector(*results), None
    elif ast[0] == 'get':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.get(arg1, arg2), None
    elif ast[0] == 'put':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        arg3, _ = evaluate_program(ast[3], env)
        return primitives.put(arg1, arg2, arg3), None
    elif ast[0] == 'first':
        arg, _ = evaluate_program(ast[1], env)
        return primitives.first(arg), None
    elif ast[0] == 'second':
        arg, _ = evaluate_program(ast[1], env)
        return primitives.second(arg), None
    elif ast[0] == 'last':
        arg, _ = evaluate_program(ast[1], env)
        return primitives.last(arg), None
    elif ast[0] == 'append':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.append(arg1, arg2), None
    elif ast[0] == 'cons':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.cons(arg1, arg2), None
    elif ast[0] == 'conj':
        arg1, _ = evaluate_program(ast[1], env)
        arg2 = [evaluate_program(ast[i])[0] for i in range(2, len(ast))]
        return primitives.conj(arg1, arg2), None
    elif ast[0] == 'hash-map':
        results = [evaluate_program(ast[i], env)[0] for i in range(1, len(ast))]
        return primitives.hash_map(*results), None
    elif ast[0] == 'normal':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.normal(arg1, arg2), None
    elif ast[0] == 'beta':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.beta(arg1, arg2), None
    elif ast[0] == 'exponential':
        arg1, _ = evaluate_program(ast[1], env)
        return primitives.exponential(arg1), None
    elif ast[0] == 'uniform':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.uniform(arg1, arg2), None
    elif ast[0] == 'discrete':
        arg1, _ = evaluate_program(ast[1], env)
        return primitives.discrete(arg1), None
    elif ast[0] == 'sample':
        arg1, _ = evaluate_program(ast[1], env)
        return arg1.sample(), None
    elif ast[0] == 'let':
        v1 = ast[1][0]
        e1_val, _ = evaluate_program(ast[1][1], env)
        env[v1] = e1_val
        return evaluate_program(ast[2], env)
    elif ast[0] == 'rest':
        arg1, _ = evaluate_program(ast[1], env)
        return primitives.rest(arg1), None
    elif ast[0] == 'mat-transpose':
        arg1, _ = evaluate_program(ast[1], env)
        return primitives.mat_transpose(arg1), None
    elif ast[0] == 'mat-tanh':
        arg1, _ = evaluate_program(ast[1], env)
        return primitives.mat_tanh(arg1), None
    elif ast[0] == 'mat-add':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.mat_add(arg1, arg2), None
    elif ast[0] == 'mat-mul':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        return primitives.mat_mul(arg1, arg2), None
    elif ast[0] == 'mat-repmat':
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        arg3, _ = evaluate_program(ast[3], env)
        return primitives.mat_repmat(arg1, arg2, arg3), None
    elif ast[0] == 'if':
        e1_val, _ = evaluate_program(ast[1], env)
        if e1_val:
            return evaluate_program(ast[2], env)
        else:
            return evaluate_program(ast[3], env)
    elif ast[0] in ['<', '>', '<=', '>=', '==']:
        arg1, _ = evaluate_program(ast[1], env)
        arg2, _ = evaluate_program(ast[2], env)
        if ast[0] == '<':
            return arg1 < arg2, None
        elif ast[0] == '>':
            return arg1 > arg2, None
        elif ast[0] == '<=':
            return arg1 <= arg2, None
        elif ast[0] == '>=':
            return arg1 >= arg2, None
        elif ast[0] == '==':
            return arg1 == arg2, None
    elif ast[0] == 'observe':
        return None, None
    else:
        print(env)
        raise Exception(ast)

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)[0]



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


if __name__ == '__main__':

    run_deterministic_tests()
    run_probabilistic_tests()


    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print(ast)
        print('\n\n\nSample of prior of program {}:'.format(i))
        acc = []
        for _ in range(1000):
            acc.append(evaluate_program(ast)[0])
        if i == 4:
            with open(str(i) + ".npy", 'wb') as f:
                for j in range(4):
                    part_acc = []
                    for k in range(1000):
                        part_acc.append(acc[k][j].numpy())
                    print(part_acc)
                    np.save(f, np.stack(part_acc))
        else:
            acc = torch.stack(acc)
            with open(str(i) + ".npy", 'wb') as f:
                np.save(f, acc.numpy())
