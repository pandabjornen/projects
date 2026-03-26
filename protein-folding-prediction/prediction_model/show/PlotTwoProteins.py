
import numpy as np
import matplotlib.pyplot as plt


def PlotTwoProteins(CaCoordsPred, CaCoordsCorrect, title): 
    
    fig = plt.figure(figsize=(10, 10))
    plt.title(title)
    ax = fig.add_subplot(111, projection='3d')

    for j, CaCoords in enumerate([CaCoordsPred, CaCoordsCorrect]): 
        if j == 0: 
            color = "red"
        else: 
            color = "green"
        for coords in CaCoords:

            ax.scatter(
                coords[0], 
                coords[1], 
                coords[2],
                color= color,
                marker='o'
            )
        for i in range(len(CaCoords)-1):

            ax.plot(
                [CaCoords[i][0], CaCoords[i+1][0]],
                [CaCoords[i][1], CaCoords[i+1][1]],
                [CaCoords[i][2], CaCoords[i+1][2]],
                color=color, linewidth=0.5, alpha=0.7
            )


    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_zlabel('z [Å]')
 
    plt.title('Protein Structure', fontsize = 30)
    ax.view_init(elev=30, azim=30)


    plt.show()