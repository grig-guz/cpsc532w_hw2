import torch
import math
import torch.distributions as dist


def plus(arg1, arg2):
    return arg1 + arg2

def sqrt(arg):
    return torch.sqrt(arg)

def product(arg1, arg2):
    return arg1 * arg2

def div(arg1, arg2):
    return arg1 / arg2

def vector(*arglist):
    for arg in arglist:
        if not isinstance(arg, torch.Tensor):
            return list(arglist)
    try:
        return torch.stack(arglist, dim=0)
    except Exception:
        return list(arglist)

def append(arg1, arg2):
    if isinstance(arg1, list):
        arg1.append(arg2)
    elif isinstance(arg1, torch.Tensor):
        if arg1.dim() == 0:
            arg1 = arg1.unsqueeze(dim=0)
        if arg2.dim() == 0:
            arg2 = arg2.unsqueeze(dim=0)
        arg1 = torch.cat([arg1, arg2])
    return arg1

def hash_map(*arglist):
    hmap = {}
    for i in range(0, len(arglist), 2):
        key = arglist[i]
        if isinstance(key, torch.Tensor):
            key = float(key)
        hmap[key] = arglist[i+1]
    return hmap

def get(arg1, arg2):
    if isinstance(arg1, dict):
        return get_hashmap(arg1, arg2)
    return arg1[int(arg2)]

def get_hashmap(arg1, arg2):
    return arg1[float(arg2)]

def put(arg1, arg2, arg3):
    if isinstance(arg1, dict):
        return put_hashmap(arg1, arg2, arg3)
    arg1[int(arg2)] = arg3
    return arg1

def put_hashmap(arg1, arg2, arg3):
    key = float(arg2)
    arg1[key] = arg3
    return arg1

def first(arg1):
    return get(arg1, 0)

def second(arg1):
    return get(arg1, 1)

def rest(arg1):
    return arg1[1:]

def last(arg1):
    return get(arg1, -1)

def conj(arg1, arg2):
    return get(arg1, -1)

def cons(arg1, arg2):
    if isinstance(arg2, list):
        arg2.insert(0, arg1)
    elif isinstance(arg2, torch.Tensor):
        if arg1.dim() == 0:
            arg1 = arg1.unsqueeze(dim=0)
        if arg2.dim() == 0:
            arg2 = arg2.unsqueeze(dim=0)
        arg2 = torch.cat([arg1, arg2])
    return arg2

def normal(arg1, arg2):
    return dist.Normal(arg1, arg2)

def exponential(arg1):
    return dist.exponential.Exponential(arg1)

def beta(arg1, arg2):
    return dist.beta.Beta(arg1, arg2)

def uniform(arg1, arg2):
    return dist.uniform.Uniform(arg1, arg2)

def discrete(arg1):
    return dist.categorical.Categorical(arg1)

def mat_transpose(arg1):
    return arg1.T

def mat_tanh(arg1):
    return torch.tanh(arg1)

def mat_add(arg1, arg2):
    return arg1 + arg2

def mat_mul(arg1, arg2):
    return torch.matmul(arg1, arg2)

def mat_repmat(arg1, arg2, arg3):
    return arg1.repeat(int(arg2), int(arg3))
