#!/usr/bin/env python3
import numpy as np
import rbdl

# Cargar el modelo desde tu URDF
modelo = rbdl.loadModel('/home/atangana09/proyecto/src/robot/universal_robot/urdf/myrobot.urdf')

# Grados de libertad
ndof = modelo.q_size

# Vectores articulares
q = np.zeros(ndof)        # posición articular
dq = np.ones(ndof) * 0.1  # velocidad articular
ddq_zero = np.zeros(ndof)
dq_zero = np.zeros(ndof)

# Vectores para guardar resultados
tau_g = np.zeros(ndof)
tau_coriolis = np.zeros(ndof)
tau_temp = np.zeros(ndof)

# --- 1. Vector de gravedad g(q) ---
rbdl.InverseDynamics(modelo, q, dq_zero, ddq_zero, tau_g)
print("Vector de gravedad g(q):")
print(np.round(tau_g, 4))

# --- 2. Vector de Coriolis y centrífuga ---
rbdl.InverseDynamics(modelo, q, dq, ddq_zero, tau_coriolis)
c = tau_coriolis - tau_g
print("\nVector de Coriolis y centrífuga c(q,dq):")
print(np.round(c, 4))

# --- 3. Matriz de inercia M(q) ---
M = np.zeros((ndof, ndof))
for i in range(ndof):
    ei = np.zeros(ndof)
    ei[i] = 1.0
    rbdl.InverseDynamics(modelo, q, dq_zero, ei, tau_temp)
    M[:, i] = tau_temp - tau_g

print("\nMatriz de inercia M(q):")
print(np.round(M, 4))
