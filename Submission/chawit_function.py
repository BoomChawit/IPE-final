import numpy as np

def distance(p_lon,p_la,c_lon,c_la):
    forx = (p_lon*np.cos(p_la/180*np.pi)) - (c_lon*np.cos(c_la/180*np.pi))
    x = 40075160/360*(forx)
    y = 40008000/360*(p_la - c_la)
    d = (x**2 + y**2)**0.5
    return d

def users(data1,data2):
    plon = data1['longitude']
    pla = data1['latitude']
    puser = data1['users']
    clon = data2['longitude']
    cla = data2['latitude']
    
    mean_users = []
    for i,j in zip(clon,cla):
        z=0;a=0
        for m,n,p in zip(plon,pla,puser):
            if distance(m,n,i,j) < 200000:
                a = a + p
                z = z + 1
        mean_users.append(a/z)
    return mean_users
