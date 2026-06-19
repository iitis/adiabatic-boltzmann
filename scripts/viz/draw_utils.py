import math

import dwave_networkx as dnx

def which_edge(edg, edges):

    for v, eL in edges.items():
        if edg in eL or edg[::-1] in eL:
            return v
    return None

node_colormap = lambda v: 'magenta' if v in ['in2', 'out'] else \
                        'red' if v.startswith('in1') else \
                        'blue' if v.startswith('c_') else \
                        'green' if v.startswith('enable') else \
                        'grey' if v.startswith('a_') else None
def which_node(node, nodes):
    for v, vL in nodes.items():
        if node in vL: 
            return v
    return None


def arcTdot(a, b, color): # return the string for make an arc
    return "    \"{}\" -- \"{}\"[penwidth = 2.5, color={}]".format(a,b,color)

def warcTdot(a, b, color, weight): # return the string for make an arc
    return "    \"{}\" -- \"{}\"[penwidth = 2.5, color={}, label=\"\", xlabel=\"{}\"]".format(a,b,color, weight)

def nodeTdot(a, color, coord): # return the string for make a node
    return "    \"{}\" [style=filled, fillcolor={}, label=\"\",shape=circle,height=0.25,width=0.25,pos=\"{},{}!\"]".format(a, color, coord[0], coord[1])

def wnodeTdot(a, color, weight, coord): # return the string for make a node
    return "    \"{}\" [style=filled, fillcolor={}, label=\"\", xlabel=\"{}\", forcelabels=true, shape=circle,height=0.25,width=0.25,pos=\"{},{}!\"]".format(a, color, weight, coord[0], coord[1])


def graph_2_dot(file, graph, real_graph, model={}, qbt_values={}, m=16, label=''):
    map_arcs = dnx.pegasus_graph(m, coordinates=True, nice_coordinates=True)
    map_nodes = dnx.pegasus_layout(map_arcs)
    
    def c(a):
        return dnx.pegasus_coordinates(m).nice_to_linear(a)
    print("graph G{outputorder=edgesfirst;", file=file);

    resize = math.sqrt(len(map_nodes))*2 #heuristic for better looking
    for a in map_arcs:
        for b in map_arcs[a]:
            a_new = c(a)
            b_new = c(b)
            if a_new<b_new: 
                edge = tuple([a_new,b_new])
                
                #v = which_edge(edge, graph['edges']) if graph else None
                color = 'blue'
                if model and v:
                    print(warcTdot(a_new, b_new, color, model['couplings'][edge]), file=file)
                else:
                    print(arcTdot(a_new, b_new, color), file=file);     

    for a in map_nodes:
        a_new = c(a)
        
        v = which_node(a_new, graph['nodes']) if graph else None
        color = node_colormap(v) if v else 'red' if a_new not in real_graph['nodes'] else 'lightgrey'

        coord=[map_nodes[a][0]*resize, map_nodes[a][1]*resize]
        if qbt_values:
            if a_new in qbt_values.keys():
                print(wnodeTdot(a_new, color, qbt_values[a_new], coord), file=file);
            else:
                print(nodeTdot(a_new, 'lightgrey', coord), file=file);
        elif model and v: 
            print(wnodeTdot(a_new, color, model['biases'][a_new], coord), file=file);
        else:
            print(nodeTdot(a_new, color, coord), file=file);
    print("}", file=file);
