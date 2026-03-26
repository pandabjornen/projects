import torch
from torch import nn

# Materiallista för 2D-bildklassificering
"""
                    Klass                                           |            Förklaring
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
Conv2d(in_channels, out_channels, kernel_size, stride, padding)     |            2D konvolutionellt lager
ReLU()                                                              |            Aktiveringsfunktion
MaxPool2d(kernel_size, stride):                                     |            Skalar ned bilderna ([kernel_size = 2, stride =2] => varje maxpool halverar höjd och bredd på bild)
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
Linear(in_features, out_features):                                  |            Ett fullt anslutet lager där input datan behöver vara endimensionell i bemärkelsen att en bild har blivit en vektor.
Flatten():                                                          |            Högre ordningens tensor -> endim tensor (vektor)
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
Dropout(p):                                                         |            ?
BatchNorm2d(num_features):                                          |            ?
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
"""

# Materiallista för 3D-videoklassificering
"""
Conv3d(in_channels, out_channels, kernel_size, stride, padding)
ReLU(): 
MaxPool3d(kernel_size, stride): 
Linear(in_features, out_features): 
Flatten():
Dropout(p): 
BatchNorm3d(num_features): 
"""

class TinyVGG(nn.Module):
    def __init__(self, inställningar):
        super(TinyVGG, self).__init__()

        in_channels = inställningar['in_channels']
        stride = inställningar['stride']
        padding = inställningar['padding']
        noder = inställningar['noder']
        klasser = inställningar['klasser']
        kernel_size = inställningar['filter_storlek']
        kernel_size_maxpool = inställningar['filter_storlek_maxpool']
   
        höjd = int(inställningar['höjd'])   
        bredd = int(inställningar['bredd'])

   
        self.mönster = nn.Sequential(
            nn.Conv2d(in_channels, noder, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(noder, noder, kernel_size, stride, padding),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size_maxpool),
            nn.Conv2d(noder, noder, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(noder, noder, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size_maxpool),
        )
        self.klassificerare = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(noder * höjd//4 * bredd//4), klasser), 
        )

    def forward(self, x):
        x = self.mönster(x)
        x = self.klassificerare(x)
        return x



class modell3D(nn.Module):
    def __init__(self, inställningar):
        super(modell3D, self).__init__()

        #params: 
        in_channels = inställningar['in_channels']  
        stride = inställningar['stride_3D']
        stride_mp = inställningar['stride_mp_3D']
        padding = inställningar['padding']
        noder = inställningar['noder']
        klasser = inställningar['klasser']
        ks_conv = inställningar['filter_storlek_conv_3D']
        ks_maxpool = inställningar['filter_storlek_maxpool_3D']
        höjd = int(inställningar['höjd_3D'])   
        bredd = int(inställningar['bredd_3D'])
        djup = (inställningar['djup_3D'])


        self.mönster = nn.Sequential(
            nn.Conv3d(in_channels, noder, ks_conv, stride, padding),
            nn.ReLU(),
            #nn.MaxPool3d(ks_maxpool),      #### verkar fungaera utan denna, jag vet inte blir fel i nn.linear med matrisstrolekarna... 
            nn.Conv3d(noder, noder, ks_conv, stride, padding),
            nn.ReLU(),
        )
        self.klassificerare = nn.Sequential(
            nn.Flatten(),
            nn.Linear(noder*höjd*bredd*djup, klasser),
        )
        
    def forward(self, x):
        x = self.mönster(x)
        x = self.klassificerare(x)
        return x

# Mappning av modellnamn till konstruktörer
modell_mappning = {
    'TinyVGG': TinyVGG,
    'modell3D': modell3D,
}

# Funktion för att hämta en modellkonstruktör baserat på namn
def get_modell(namn, inställningar):
    if namn in modell_mappning:
        return modell_mappning[namn](inställningar)
    else:
        raise ValueError(f"Modellnamnet {namn} är inte med i modellmappning, gå in och kolla i Modeller.py.")
