import scipy.io as sio
import numpy as np

def get_data(zInfo, sess, start, end, load=False, nTrials=0, ):
    
    nTrials = np.array(zInfo['zInfoGatheringInfo'][sess]['AnalogData']).shape[0]
    
    numdlPFC = 0
    for i in range(start,end):
        data = sio.loadmat("../data/LTH_processed_data/Unit%.4d.mat" % i)
            
    print(f'{end - start} neurons in this session.')
            
    # Create raster matrices for all units 
    cue1_rast = np.zeros((end - start,nTrials,600))
    region = np.zeros(end-start)
    position = np.zeros(nTrials)
            
    ind = 0
    for i in range(start,end):
        # 
        data = sio.loadmat("../data/LTH_processed_data/Unit%.4d.mat" % i)
        region[ind] = data['brain_region'].flatten()[0]
        cue1_rast[ind] = np.array(data['Cue1_MatrixRaw'][:,0:600])
        ind += 1

    for tr in range(nTrials):
        position[tr] = data['position_code'][0][tr][0][0]

    cue1_time = data['Cue1_ON'].flatten()
    
    # Extract eye data
    EyeX_cue1 = np.zeros((nTrials,600))
    EyeY_cue1 = np.zeros((nTrials,600))

    if load==True:
        EyeX_cue1 = np.load(f'../results/EyeX_s{sess}_c1.npy')
        EyeY_cue1 = np.load(f'../results/EyeY_s{sess}_c1.npy')
        print("Loaded eye data from file.")
    else:
        for i in range(nTrials):
            eyeSignal = zInfo['zInfoGatheringInfo'][sess]['AnalogData'][i]['EyeSignal']

            EyeX_cue1[i] = eyeSignal[cue1_time[i]-200:cue1_time[i]+400,0]
            EyeY_cue1[i] = eyeSignal[cue1_time[i]-200:cue1_time[i]+400,1]

            np.save(f'../results/EyeX_s{sess}_c1.npy', EyeX_cue1)
            np.save(f'../results/EyeY_s{sess}_c1.npy', EyeY_cue1)
            
    return cue1_rast, position - 90, region, EyeX_cue1, EyeY_cue1