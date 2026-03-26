
import numpy as np
import matplotlib.pyplot as plt


def PlotOneProtein(indexOfProteinToPlot, CaCoords, AminoAcidsIndicies): 
    
    aa_to_index = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWYX')}
    index_to_aa = {i: aa for aa, i in aa_to_index.items()}
    AAIndices = AminoAcidsIndicies[indexOfProteinToPlot]
    indices = sorted(set(AAIndices.tolist()))

    distinct_colors = ['#FF0000','#00FF00','#0000FF','#FFFF00','#FF00FF','#00FFFF','#FF8000','#8000FF','#0080FF','#FF0080','#80FF00','#00FF80','#FF4000','#8000C0','#804000','#408000','#004080','#C00000','#00C000','#0000C0', '#C0C000', '#C000C0', '#00C0C0', '#000000', '#808080', '#FFFFFF'  ]

    color_map = {idx: distinct_colors[i % len(distinct_colors)] for i, idx in enumerate(indices)}

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    for i, aa_idx in enumerate(AAIndices):
        if aa_idx == 0:
            continue
        ax.scatter(
            CaCoords[indexOfProteinToPlot][i][0], 
            CaCoords[indexOfProteinToPlot][i][1], 
            CaCoords[indexOfProteinToPlot][i][2],
            color=color_map[aa_idx],
            marker='o'
        )
    for i in range(len(AAIndices)-1):
        if AAIndices[i] == 0 or AAIndices[i+1] == 0:
            continue
        ax.plot(
            [CaCoords[indexOfProteinToPlot][i][0], CaCoords[indexOfProteinToPlot][i+1][0]],
            [CaCoords[indexOfProteinToPlot][i][1], CaCoords[indexOfProteinToPlot][i+1][1]],
            [CaCoords[indexOfProteinToPlot][i][2], CaCoords[indexOfProteinToPlot][i+1][2]],
            color='black', linewidth=0.5, alpha=0.7
        )


    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_zlabel('z [Å]')

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color_map[idx], markersize=10,
                                label=f'{index_to_aa.get(idx, "?")} ({idx})') 
                    for idx in indices]

    ax.legend(handles=legend_elements, loc='upper right')
    plt.title('Protein Structure', fontsize = 30)
    ax.view_init(elev=30, azim=30)


    plt.show()