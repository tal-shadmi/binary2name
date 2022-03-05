

class Vertex:
    def __init__(self, baddr: int, instructions: str, constraint: list = []):
        self.baddr = baddr
        self.instructions = instructions
        self.constraint = constraint
    
    # we define uniqueness by address only
    def __eq__(self, other):
        assert(isinstance(other, Vertex))
        return self.baddr == other.baddr

    def __str__(self):
        return f'{{ "block_addr": {self.baddr}, "instructions": "{self.instructions}", "constraints": {self.constraint} }}'

class Edge:
    def __init__(self, source: int, dest: int):
        self.source = source
        self.dest = dest

    def __eq__(self, other):
        assert(isinstance(other, Edge))
        return (self.source == other.source and self.dest == other.dest)

    def __str__(self):
        return f'{{ "src": {self.source}, "dst": {self.dest} }}'

        


class SymGraph: # TODO: sanity check, when graph is done, vertices.keys() length is same as edges.keys()
    def __init__(self, root: Vertex, func_name: str="unknown_function"):
        self.root = root
        self.vertices = {}
        self.edges = {}
        self.addVertex(root)
        self.func_name = func_name

    def addVertex(self, vertex: Vertex):
        vertex.constraint = list(filter(None, vertex.constraint))
        
        if vertex.baddr in self.vertices:
            for constraint in self.vertices[vertex.baddr].constraint:
                if constraint not in vertex.constraint:
                    vertex.constraint.append(constraint)
        
        self.vertices[vertex.baddr] = vertex

        if (vertex.baddr not in self.edges.keys()):
            self.edges[vertex.baddr] = []

    def addEdge(self, edge: Edge):
        assert(edge.source in self.vertices.keys() and edge.source in self.edges.keys())
        assert(edge.dest in self.vertices.keys() and edge.dest in self.edges.keys())

        if (edge not in self.edges[edge.source]):
            self.edges[edge.source].append(edge)

    #TODO: redo the printing!
    def __str__(self):
        res = f'{{ "func_name": "{self.func_name}",'
        res += f'"GNN_DATA": {{ '
        res += f'"nodes": [ '
        res += ', '.join([str(v) for v in list(self.vertices.values())])

        res += f' ], "edges": [ '
        all_edges = [item for sublist in self.edges.values() for item in sublist]
        res += ', '.join([str(e) for e in all_edges])
        
        res += f' ] }} }}'
        return res

        
