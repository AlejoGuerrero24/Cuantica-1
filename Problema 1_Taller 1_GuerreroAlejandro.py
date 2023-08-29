import numpy as np
import matplotlib.pyplot as plt

"""Parte a y b del ejercicio"""

# Constantes
h = 6.626e-34  
c = 3.0e8     
k = 1.381e-23   

# Definir temperaturas
T = [3000,5000,8000,10000]  # en Kelvin

# Rango de longitudes de onda
L = np.linspace(1e-9, 3e-6, 100000)  # en metros

# Función para la densidad de energía por unidad de frecuencia
def densidad_energia(l, t):
    densidad = ((8 * np.pi *h) /(l**3)) * (1 /(np.exp((h *c)/(l*k*t)) - 1))
    return densidad

def longitud_y_espectro():
    for t in T:
        energia = densidad_energia(L, t)
        posicion = list(energia).index((max(energia)))
        longitud = round(((L[posicion])*1e9),2)
        
        if (420 < longitud) and (longitud < 720):
            luz = "Espectro Visible"
        if (longitud > 720) and (longitud < 2500):
            luz = "Espectro Infrarojo"
        if (longitud) > 10 and (longitud < 420):
            luz = "Espectro Ultra Violeta"
        print("\nPico para una Temperatura "+str(t)+" K: "+str(longitud)+" nm. Por lo que esta en el rango de: "+luz)
        

longitud_y_espectro()
        

# Crear la gráfica
plt.figure(figsize=(10, 8))

for t in T:
    energia = densidad_energia(L, t)
    plt.plot(L * 1e9, energia, label=f'{t} K')

plt.xlabel('Longitud de Onda (nm)') 
plt.ylabel('Densidad de Energía (J/Hz*m³)')
plt.title('Densidad de Energía por Vs Longitud de onda')
plt.legend()
plt.grid(True)
plt.show()

"""Parte c del ejercicio"""

F = np.linspace(1e12, 1e15, 100000) # en Hz

# Ponemos las formulas de cada caso

def densidad_energia_f(f, t):
    return ((8 * np.pi *h*(f**3))/(c**3)) * (1 /(np.exp((h*f)/(k*t)) - 1))

def wien(f,t):
    return ((8*np.pi*h*(f**3))/(c**3))*(1/(np.exp((h*f)/(k*t))))

def R_J(f,t):
    return (8*np.pi*k*t*(f**2))/(c**3)

planck = list(densidad_energia_f(F, 5000))

wien = list(wien(F, 5000))

R = list(R_J(F, 5000))

plt.figure(figsize=(10, 8)) #Creamos la grafica

formula = ["Planck","Wien","Ryleigh"]

plt.plot(F*1e-12, planck, label=formula[0])
plt.plot(F*1e-12, wien, label=formula[1])
plt.plot(F*1e-12, R, label=formula[2])
    
plt.xlabel('Frecuencia (THz)') 
plt.ylabel('Densidad de Energía (J/Hz*m³)')
plt.title('Densidad de Energía vs Frecuencia a 5000K')
plt.xlim(0,1000)
plt.ylim(0,0.0000000000000012)
plt.legend()
plt.grid(True)
plt.show()

# Vamos a hacer la comparativa entre las curvas de Rayleigh y wien con respecto a la de planck

suma_p=sum(planck)

suma_w=sum(wien)

suma_r=sum(R)

print("\nEl porcentaje de diferencia entre la curva de Planck y la de Wien es de: ", round(((suma_p-suma_w)/suma_p)*100,2),"%")

print("\nEl porcentaje de diferencia entre la curva de Planck y la de Rayleigh es de: ", round((abs(suma_p-suma_r)/suma_p)*100,2),"%")

# Se observa que en general las curvas son disparejas con respecto a al de planck, una mas que otra.

suma_p1=sum(planck[0:250])

suma_p2=sum(planck[40000:-1])

suma_w1=sum(wien[40000:-1])

suma_r1=sum(R[0:250])

print("\nEl porcentaje de diferencia entre la curva de Planck y la de Wien cuando se toman frecuencias altas unicamente es: ", round(((suma_p2-suma_w1)/suma_p2)*100,2),"%")

print("\nEl porcentaje de diferencia entre la curva de Planck y la de Rayleigh cuando se toman frecuencias muy bajas unicamente es: ", round((abs(suma_p1-suma_r1)/suma_p1)*100,2),"%")

# Sin embargo, cuando tomamos valores los cuales las curvas de Rayleigh y wien se acoplan mejor a lo observado en el laboratorio
# Vemos que el porcentaje de diferencia con respecto a la curva de planck cae drasticamente lo que las hace mas similares en
# Este tipo de rangos.
