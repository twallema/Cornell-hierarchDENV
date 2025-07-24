"""
This script simulates the model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, Bento Lab, Cornell University CVM Public Health. All Rights Reserved."

from datetime import datetime
import matplotlib.pyplot as plt
from hierarchDENV.utils import initialise_model

# settings
serotypes = False
uf = 'MG'

# initialise
strains = 4 if serotypes is True else 1
model = initialise_model(strains=strains, uf=uf)

# simulate
simout = model.sim([datetime(2024,10,1), datetime(2025,10,1)])

# visualise
fig,ax=plt.subplots()
#ax.plot(simout['date'], simout['I'], label='I (prev.)')
ax.plot(simout['date'], simout['I_inc'], label='I (observed inc.)')
ax.legend()
plt.show()
plt.close()