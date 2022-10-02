import numpy as np
import json
# import nmodels as n
import test as test


def get_json_info(jsonfile):
    with open(jsonfile) as f:
        data = json.load(f)
    coords = data['coordinates']
    masses = data['simMasses']
    #connections = data['connections']
    return coords, masses

# 4069,4088
length=18 # length of the sequence
p = np.array((6490,3836,4088,4944,5276,5577,5752,6051,6116,6157,6223,6291,6358,6424,6531,6598,6663,6729),dtype=int) #select the whole sequence decolor it.
# search for long TTTTTTTTTT sequence and mark their 3to5. paste them above
p-=1
handles=np.zeros(len(p)*18)

for i in range((len(p))):

    handles[i*length:(i+1)*length]=np.linspace(p[i],p[i]-length+1,length)

p-=length
# print(p)


N= 16
target_coords, masses = get_json_info('octahedron.json')
target_coords = np.asarray(target_coords)
cg = test.Coarse_Grainer(target_coords, N, kstart=True, distance_cutoff=200, votes=0, ignore=handles, fix_handles=p, max_radius_decimal=0.6)
adjusted_coords = cg.target_coords
new_positions = cg.coarse_grainer(adjusted_coords, steps=1, timestep=0.002, hmc_steps=2)
cg.network_export(f'ico_3p_{N}_nh_{N}') # generates two files! the network description with particles and masses, and an index file which describes which particles in the original system are represented by each particle in the coarse grained system