import angr
from angr.project import Project
from angr.sim_state import SimState
import networkx as nx
import matplotlib.pyplot as plt
from networkx import Graph

def show_constraint_location(proj: Project, state: SimState):
    curr = state.history
    while curr.addr:
        print("block: ")
        proj.factory.block(curr.addr).pp()
        print("constraints:", curr.recent_constraints)
        curr = curr.parent

proj = angr.Project('complex_constraints/dummy.out')
fobj = proj.loader.find_symbol('checkme')
state = proj.factory.call_state(fobj.rebased_addr)
simgr = proj.factory.simgr(state)
cfg = proj.analyses.CFGFast()
fun_obj = cfg.kb.functions[fobj.rebased_addr]
nx.draw(fun_obj.transition_graph)
plt.show()




simgr.use_technique(angr.exploration_techniques.LoopSeer(cfg=cfg, bound=2))

#generating the states
simgr.run()
first = simgr.deadended[0]
second = simgr.deadended[1]

#show the code of each path IN REVERSE ORDER:
print('FOR FIRST PATH:')
show_constraint_location(proj, first)
print('FOR SECOND PATH:')
show_constraint_location(proj, second)



