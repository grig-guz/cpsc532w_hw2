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

def sample_from_prior(graph, env):
    top_sort = topological_sort(graph['V'], graph['A'])
    for x in top_sort:
        if x in graph['Y']:
            env['buffer'][x] = graph['Y'][x]
        else:
            env['buffer'][x] = env['det_eval'](graph['P'][x], env)
