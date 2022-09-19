import numpy as np

def find_cs_sn(thetaA,phiA,thetaB,phiB):
    '''
    theta is the angle 
    between the meridian crossing point A
    and the geodesic connecting A and B
    
    this function return cos and sin of theta
    '''
    # O being north pole
    AO = np.pi/2 - thetaA
    BO = np.pi/2 - thetaB
    dphi = phiB-phiA
    # Spherical law of cosine on AOB
    cos_AB = np.cos(BO)*np.cos(AO)+np.sin(BO)*np.sin(AO)*np.cos(dphi)
    sin_AB = np.sqrt(1-cos_AB**2)
    # spherical law of sine on triangle AOB
    SN = np.sin(BO)*np.sin(dphi)/sin_AB
    CS = np.sign(thetaB-thetaA)*np.sqrt(1-SN**2)
    return CS,SN

def missing_cs_sn(od):
    xc = np.deg2rad(np.array(od._ds.XC))
    yc = np.deg2rad(np.array(od._ds.YC))
    cs = np.zeros_like(xc)
    sn = np.zeros_like(xc)
    cs[0],sn[0] = find_cs_sn(
        yc[0],xc[0],
        yc[1],xc[1]
    )
    cs[-1],sn[-1] = find_cs_sn(
        yc[-2],xc[-2],
        yc[-1],xc[-1]
    )
    cs[1:-1],sn[1:-1] = find_cs_sn(
        yc[:-2],xc[:-2],
        yc[2:],xc[2:]
    )
    od._ds['CS'] = od._ds['XC']
    od._ds['CS'].values = cs
    
    od._ds['SN'] = od._ds['XC']
    od._ds['SN'].values = sn