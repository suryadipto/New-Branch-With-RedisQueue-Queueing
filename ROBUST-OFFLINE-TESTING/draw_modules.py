import networkx as nx

def draw_module(filepath):
    module = nx.read_graphml(filepath)
    print('Total no. of nodes:')
    print(len(module.nodes))
    print('Total no. of edges:')
    print(len(module.edges))
    seeds=0
    nonseeds=0
    node_colors = []
    for _, data in module.nodes(data=True):
        if data['isSeed']:
            node_colors.append('red')
            seeds+=1
        else:
            node_colors.append('orange')
            nonseeds+=1
    print('=======================')
    print('No. of seeds')
    print(seeds)
    print('No. of non-seeds:')
    print(nonseeds)
    nx.draw_networkx(module,node_color=node_colors)

draw_module('APID_UNIPROT_alpha-beta-n-tau-studybiasscores-gamma-default.graphml')
# draw_module('BioGRID_UNIPROT_alpha-beta-n-tau-studybiasscores-gamma-default.graphml')
# draw_module('customGRPAHMLhprd_ENTREZ_alpha-beta-n-tau-gamma-default.graphml')
# draw_module('customGRPAHMLhprd_GENESYMBOL_alpha-beta-n-tau-gamma-default.graphml')
# draw_module('HPRD_UNIPROT_alpha-beta-n-tau-studybiasscores-gamma-default.graphml')
# draw_module('STRING_UNIPROT_alpha-beta-n-tau-studybiasscores-gamma-default.graphml')
