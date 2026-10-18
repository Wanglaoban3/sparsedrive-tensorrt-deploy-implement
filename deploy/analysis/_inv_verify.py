# 按 sp_modelnode.cpp mat4_inv 的 C 一维扁平下标逐句镜像, 对拍 np.linalg.inv
import numpy as np

def mat4_mul(a, b):
    o = np.zeros(16)
    for r in range(4):
        for c in range(4):
            v = 0.0
            for k in range(4):
                v += a[r*4+k] * b[k*4+c]
            o[r*4+c] = v
    return o

def mat4_inv(m):
    o = np.zeros(16)
    tx, ty, tz = m[3], m[7], m[11]
    o[0] = m[0]; o[1] = m[4]; o[2] = m[8]
    o[3] = -(m[0]*tx + m[4]*ty + m[8]*tz)
    o[4] = m[1]; o[5] = m[5]; o[6] = m[9]
    o[7] = -(m[1]*tx + m[5]*ty + m[9]*tz)
    o[8] = m[2]; o[9] = m[6]; o[10] = m[10]
    o[11] = -(m[2]*tx + m[6]*ty + m[10]*tz)
    o[12] = 0.0; o[13] = 0.0; o[14] = 0.0
    o[15] = 1.0
    return o

rng = np.random.default_rng(0)
worst = 0.0
for i in range(200):
    ax, ay, az = rng.normal(size=3) * 3
    cx = np.cos(ax); sx = np.sin(ax)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    cy = rng.normal() * 3
    cyv = rng.normal(); Cy = np.array([[np.cos(cyv),0,np.sin(cyv)],[0,1,0],[-np.sin(cyv),0,np.cos(cyv)]])
    czv = rng.normal(); Cz = np.array([[np.cos(czv),-np.sin(czv),0],[np.sin(czv),np.cos(czv),0],[0,0,1]])
    R = Cz @ Cy @ Rx
    t = rng.normal(size=3) * 500
    m = np.eye(4); m[:3,:3] = R; m[:3,3] = t
    m1 = m.reshape(-1)  # 行主序扁平
    inv = mat4_inv(m1).reshape(4,4)
    ref = np.linalg.inv(m)
    worst = max(worst, np.abs(inv - ref).max())
print("random rigid  worst |diff| vs np.linalg.inv =", worst)

# 再对拍复合口径: tmat = inv(l2g_cur) @ l2g_prev 底行必须 [0,0,0,1]
m = np.eye(4); m[:3,:3] = R; m[:3,3] = t
m1 = m.reshape(-1)
tm = mat4_mul(mat4_inv(m1), m1.copy())
print("inv(m)@m bottom row =", tm[12:16])
assert worst < 1e-9 and np.allclose(tm[12:16], [0,0,0,1])
print("OK")
