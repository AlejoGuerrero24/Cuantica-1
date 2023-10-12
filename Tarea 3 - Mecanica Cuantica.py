import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-5,5,1000)
h = (5-(-5))/(len(X)-1)
b = (h**2)/12

#Potencial armonico

def V(x):
    return (x**2)/2

def R(x, e):
    res = (2)*(V(x)-e)
    return res

def numerov(X,e):
    Y = np.zeros(len(X))
    Y[0] = 0
    Y[1] = 1 * (10**-5)
    i = 1
    while (i+1) < len(X):
        Y[i+1] = ((2*(1+(b*5*R(X[i], e)))*Y[i])-((1-(b*R(X[i-1], e)))*Y[i-1]))/(1-(b*R(X[i+1], e)))
        i +=1 
    return Y

def valores_propios(X, e):
    dE = 0.001
    Y_e = numerov(X, e)
    Y_de = numerov(X, e+dE)
    res = Y_e[-1]/Y_de[-1]
    return res

def solucion(X):
    e = 0.498
    i = 0
    valores = []
    funciones = []
    while len(valores)< 6:
        centinela = False
        while not centinela:
            Y = numerov(X, e)
            Val = valores_propios(X, e)
    
            if Val < 0:
                centinela = True
            else:
                e += 0.001
            i +=1
        valores += [e]
        funciones += [Y]
        e += 0.995
    return funciones, valores

def solucion_g(X, E):
    funciones = []
    for e in E:
        Y = numerov(X, e)
        funciones += [Y]
    return funciones


Energias = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

funciones, valores = solucion(X)  
Y1 = funciones[0]
Y2 = funciones[1]
Y3 = funciones[2]
Y4 = funciones[3]
Y5= funciones[4]
Y6 = funciones[5]

fig=plt.figure(figsize=(5.25, 5.25)) 
ax=fig.add_subplot(111)
plt.plot(X,Y1/np.sqrt(np.sum(Y1**2)*h/1.66))
plt.plot(X,Y2/np.sqrt(np.sum(Y2**2)*h/2.25))
plt.plot(X,Y3/np.sqrt(np.sum(Y3**2)*h/2.5))
plt.plot(X,Y4/np.sqrt(np.sum(Y4**2)*h/2.75))
plt.plot(X,Y5/np.sqrt(np.sum(Y5**2)*h/2.85))
plt.plot(X,Y6/np.sqrt(np.sum(Y6**2)*h/3))
plt.xlabel('X')
plt.ylabel('funcion')
plt.xlim(-5,5)
plt.legend(Energias)

#Potencial Gaussiano

Energias_I = [-9.51, -8.54, -7.62,-6.74,-5.89]
X = np.linspace(-5,5,1000)

def V_I(x):
    return -10*np.exp(-(x**2)/20)

def R_I(x, e):
    return (2)*(V_I(x)-e)
   

def metodo_de_numerov_I(X,e):
    Y = np.zeros(len(X))
    Y[0] = 0
    Y[1] = 1 * (10**-5)
    i = 1
    while (i+1) < len(X):
        Y[i+1] = ((2*(1+(b*5*R_I(X[i], e)))*Y[i])-((1-(b*R_I(X[i-1], e)))*Y[i-1]))/(1-(b*R_I(X[i+1], e)))
        i +=1 
    return Y

def solucion_I(X, E):
    funciones = []
    for e in E:
        Y = metodo_de_numerov_I(X, e)
        funciones += [Y]
    return funciones

funciones_I = solucion_I(X, Energias_I)  
Y1_I = funciones_I[0]
Y2_I = funciones_I[1]
Y3_I = funciones_I[2]
Y4_I = funciones_I[3]
Y5_I = funciones_I[4]


fig=plt.figure(figsize=(5.25, 5.25)) 
ax=fig.add_subplot(111)
plt.plot(X,Y1_I/np.sqrt(np.sum(Y1_I**2)*h))
plt.plot(X,Y2_I/np.sqrt(np.sum(Y2_I**2)*h))
plt.plot(X,Y3_I/np.sqrt(np.sum(Y3_I**2)*h))
plt.plot(X,Y4_I/np.sqrt(np.sum(Y4_I**2)*h))
plt.plot(X,Y5_I/np.sqrt(np.sum(Y5_I**2)*h))
plt.xlabel('X')
plt.ylabel('funcion')
plt.xlim(-5,5)
plt.ylim(-1,1)
plt.legend(Energias_I)

Energias_E = [-1.478,-0.163]
X = np.linspace(-5,5,1000)

#Potencial Racional

def V_E(x):
    return -4/((1+x**2)**2)

def R_E(x, e):
    return 2*(V_E(x)-e)

def metodo_de_numerov_E(X,e):
    Y = np.zeros(len(X))
    Y[0] = 0
    Y[1] = 1 * (10**-5)
    i = 1
    while (i+1) < len(X):
        Y[i+1] = ((2*(1+(b*5*R_E(X[i], e)))*Y[i])-((1-(b*R_E(X[i-1], e)))*Y[i-1]))/(1-(b*R_E(X[i+1], e)))
        i +=1 
    return Y

def solucion_E(X, E):
    funciones = []
    for e in E:
        Y = metodo_de_numerov_E(X, e)
        funciones += [Y]
    return funciones

funciones_E = solucion_E(X, Energias_E)  
Y1_I = funciones_E[0]
Y2_I = funciones_E[1]

print(h)

fig=plt.figure(figsize=(5.25, 5.25)) 
ax=fig.add_subplot(111)
plt.plot(X,Y1_I/np.sqrt(np.sum(Y1_I**2)*0.001))
plt.plot(X,Y2_I/np.sqrt(np.sum(Y2_I**2)*0.001))
plt.xlabel('X')
plt.ylabel('funcion')
plt.xlim(-5,5)
plt.ylim(-1,1)
plt.legend(Energias_E)