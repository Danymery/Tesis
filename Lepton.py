#!/usr/bin/env python
# coding: utf-8

# In[1]:


import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import math


# In[2]:


# Recibe la dirección del root. Retorna un DataFrame con la información del evento.
def Data(File_name,treename):
    # Se asegura que el nombre del archivo y la carpeta asociada al archivo sean tipo string
    if type(File_name) and type(treename) == str:
        upfile = {}
        Datos = {}
        upfile['read'] = uproot.open(File_name)
        Datos['info'] = pd.DataFrame(upfile['read'][treename].arrays())
        return Datos['info']
    else:
        print("File_name debe ser la dirección de los datos") 


# In[3]:


def Z_to_Di_Lepton(Datos,Lepton):
    
    # Comprueba que Lepton sea un string
    if type(Lepton) == str:
       
        # Número total de eventos
        total = np.size(Datos[Lepton+'_ch'])
        #print(total)

        # Contador para los índices 
        cont = 0
        
        # Verifica que sean dileptones y carga opuesta. Cuenta todos los que cumplen la condición.
        for i in range (0,total):
            if (Datos['n'+Lepton+'_e'][i]==2):
                if (np.sum(Datos[Lepton+'_ch'][i])==0):
                    cont  =  cont + 1
        
       
        # Reserva memoria para Energía, Phi, Eta, Px, Py, Pz, Pt
        Energia = np.ones((cont,2))
        Phi = np.ones((cont,2))
        Eta = np.ones((cont,2))
        Pt = np.ones((cont,2))
        
    
        cont = 0 
        
        # Almacena la Energía, Phi, Eta, Px, Py, Pz 
        for i in range (0,total):
            if (Datos['n'+Lepton+'_e'][i]==2):
                if (np.sum(Datos[Lepton+'_ch'][i])==0):
                    Pt[cont][:] = Datos[Lepton+'_pt'][i]
                    Energia[cont][:] = Datos[Lepton+'_e'][i]
                    Phi[cont][:] = Datos[Lepton+'_phi'][i]
                    Eta[cont][:] = Datos[Lepton+'_eta'][i]
                    
                    cont = cont + 1
        
        return Pt,Energia,Phi,Eta,cont
    
    else:
        return "Lepton debe ser un string."


# In[4]:


def leptonjets(Datos,Lepton,numjet):
    
    # Comprueba que Lepton sea un string
    if type(Lepton) == str:
        
        # Número total de eventos
        total = np.size(Datos[Lepton+'_ch'])
        #print(total)

        # Contador para los índices 
        cont = 0
        
        # Cuenta todos los que cumplen la condición.
        for i in range (0,total):
            if (Datos['n'+Lepton+'_e'][i]==2 and Datos['njet_e'][i]==numjet):
                if (np.sum(Datos[Lepton+'_ch'][i])==0):
                    cont  =  cont + 1
        
       
        # Reserva memoria para Energía, Phi, Eta, Px, Py, Pz, Pt
        Energia = np.ones((cont,numjet+2))
        Phi = np.ones((cont,numjet+2))
        Eta = np.ones((cont,numjet+2))
        Pt = np.ones((cont,numjet+2))
    
        cont = 0 
        
        # Almacena la Energía, Phi, Eta, Px, Py, Pz 
        for i in range (0,total):
            if (Datos['n'+Lepton+'_e'][i]==2 and Datos['njet_e'][i]==numjet):
                if (np.sum(Datos[Lepton+'_ch'][i])==0):
                
                # información muon 
                    Pt[cont][0:2] = Datos[Lepton+'_pt'][i]
                    Energia[cont][0:2] = Datos[Lepton+'_e'][i]
                    Phi[cont][0:2] = Datos[Lepton+'_phi'][i]
                    Eta[cont][0:2] = Datos[Lepton+'_eta'][i]
                    
                    # información jets
                    for j in range (0,numjet):
                        Pt[cont][j+2] = Datos['jet_pt'][i][j]
                        Energia[cont][j+2] = Datos['jet_e'][i][j]
                        Phi[cont][j+2] = Datos['jet_phi'][i][j]
                        Eta[cont][j+2] = Datos['jet_eta'][i][j]
                       
                    cont = cont + 1
        
        return Pt,Energia,Phi,Eta,cont
    
    else:
        return "Lepton debe ser un string."


# In[5]:


def Z_to_4l(Datos,lepton_uno,lepton_dos):
    
    if(type(lepton_uno)==str and type(lepton_dos)==str):
        # Número total de eventos
        # Número de electrones y muones coincide
        total = np.size(Datos[lepton_uno+'_ch'])
        #print(total)
        # Contador para los índices 
        cont = 0
         
        # Verifica que sean dileptones y carga opuesta. Cuenta todos los que cumplen la condición.
        for i in range (0,total):
            if (Datos['n'+lepton_uno+'_e'][i]==2 and Datos['n'+lepton_dos+'_e'][i]==2):
                if (np.sum(Datos[lepton_uno+'_ch'][i])==0 and np.sum(Datos[lepton_dos+'_ch'][i])==0):
                    cont  =  cont + 1
        # Reserva memoria para Energía, Phi, Eta, Px, Py, Pz, Pt
        Pt  = np.ones((cont,4))
        Energia = np.ones((cont,4))
        Phi = np.ones((cont,4))
        Eta = np.ones((cont,4))
        
        cont = 0 
        # Almacena la Energía, Phi, Eta, Px, Py, Pz 
        for i in range (0,total):
            if (Datos['n'+lepton_uno+'_e'][i]==2 and Datos['n'+lepton_dos+'_e'][i]==2):
                if (np.sum(Datos[lepton_uno+'_ch'][i])==0 and np.sum(Datos[lepton_dos+'_ch'][i])==0):
                    # Las primeras dos columnas tienen información del electron
                    Pt[cont][0:2] = Datos[lepton_uno+'_pt'][i]
                    Energia[cont][0:2] = Datos[lepton_uno+'_e'][i]
                    Phi[cont][0:2] = Datos[lepton_uno+'_phi'][i]
                    Eta[cont][0:2] = Datos[lepton_uno+'_eta'][i]
                    
                    # Las primeras dos columnas tienen información del muon
                    Pt[cont][2:4] = Datos[lepton_dos+'_pt'][i]
                    Energia[cont][2:4] = Datos[lepton_dos+'_e'][i]
                    Phi[cont][2:4] = Datos[lepton_dos+'_phi'][i]
                    Eta[cont][2:4] = Datos[lepton_dos+'_eta'][i]
                    
                    cont = cont + 1
        return Pt,Energia,Phi,Eta
    else: 
        return("Se debe ingresar el nombre del lepton como str")
    


# In[6]:


# Masa invariante formula eta phi pt
def invariant_mass(eta,phi,pt,energia):
    invariant_mass=np.ones((len(eta),1))
    for i in range(len(eta)):
        invariant_mass[i][0]=np.sqrt(2*pt[i][0]*pt[i][1]*(np.cosh(eta[i][0]-eta[i][1])-np.cos(phi[i][0]-phi[i][1])))
    return invariant_mass


# In[7]:


# total debe ser número de leptones que cumple con dos condiciones.
    # 1. Un evento que termina en dilepton.
    # 2. Un dilepton de carga opuesta.
# El valor total se obtiene de la función "Z_to_Di_Lepton". Es el sexto parámetro que devuelve la función. 

def images(cont,Energia,Eta,Phi,ruta):
    
    escalaEnergia1 = math.log(Energia[0],1.1)
    escalaEnergia2 = math.log(Energia[1],1.1)

    
    phiAxis1 = np.array([escalaEnergia1*4*np.pi/224])
    phiAxis2 = np.array([escalaEnergia2*4*np.pi/224])
    
    #print("phiAxis",phiAxis1,phiAxis2)
    
    etaAxis1 = np.array([escalaEnergia1*12/224])
    etaAxis2 = np.array([escalaEnergia2*12/224])
    
    #print("etaAxis",etaAxis1,etaAxis2)
    
    center1 = np.array([Eta[0],Phi[0]])
    center2 = np.array([Eta[1],Phi[1]])
    
    #print("center",center1,center2)
    
    fig = plt.figure()
    fig.set_size_inches(4,4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis([-6,6,-2*np.pi,2*np.pi])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    Object1 = Ellipse(xy = center1, width=etaAxis1, height=phiAxis1, angle=0.0, facecolor = 'none', edgecolor= 'black', lw = 4)
    Object2 = Ellipse(xy = center2, width=etaAxis2, height=phiAxis2, angle=0.0, facecolor = 'none', edgecolor= 'black', lw = 4)
    
    ax.add_artist(Object1)
    ax.add_artist(Object2)
    
    fig.savefig(ruta + str(cont)+ '.jpg', dpi=56)


# In[8]:


def images2(cont,Energia,Eta,Phi,lepton_uno,lepton_dos,ruta):
    
    if(lepton_dos=='muon'):
        color = 'orange'
    else:
        color = 'orange'
        
    # Lepton 1
    escalaEnergia1 = 0.25*math.log(Energia[0],1.1)
    escalaEnergia2 = 0.25*math.log(Energia[1],1.1)
    
    phiAxis1 = np.array([escalaEnergia1*4*np.pi/224])
    phiAxis2 = np.array([escalaEnergia2*4*np.pi/224])
    
    etaAxis1 = np.array([escalaEnergia1*12/224])
    etaAxis2 = np.array([escalaEnergia2*12/224])
    
    center1 = np.array([Eta[0],Phi[0]])
    center2 = np.array([Eta[1],Phi[1]])
    
    # Lepton 2
    escalaEnergia3 = 0.25*math.log(Energia[2],1.1)
    escalaEnergia4 = 0.25*math.log(Energia[3],1.1)
    
    phiAxis3 = np.array([escalaEnergia3*4*np.pi/224])
    phiAxis4 = np.array([escalaEnergia4*4*np.pi/224])
    
    etaAxis3 = np.array([escalaEnergia3*12/224])
    etaAxis4 = np.array([escalaEnergia4*12/224])
    
    center3 = np.array([Eta[2],Phi[2]])
    center4 = np.array([Eta[3],Phi[3]])
     
    
    fig = plt.figure()
    fig.set_size_inches(4,4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis([-6,6,-2*np.pi,2*np.pi])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    Object1 = Ellipse(xy = center1, width=etaAxis1, height=phiAxis1, angle=0.0, facecolor = 'none', edgecolor= 'red', lw = 4)
    Object2 = Ellipse(xy = center2, width=etaAxis2, height=phiAxis2, angle=0.0, facecolor = 'none', edgecolor= 'red', lw = 4)
    
    Object3 = Ellipse(xy = center3, width=etaAxis3, height=phiAxis3, angle=0.0, facecolor = 'none', edgecolor= color , lw = 4)
    Object4 = Ellipse(xy = center4, width=etaAxis4, height=phiAxis4, angle=0.0, facecolor = 'none', edgecolor= color, lw = 4)
    
    ax.add_artist(Object1)
    ax.add_artist(Object2)
    
    ax.add_artist(Object3)
    ax.add_artist(Object4)
    
    fig.savefig(ruta + str(cont)+ '.jpg', dpi=56)


# In[9]:


def images3(cont,Energia,Eta,Phi,ruta):
    
    # Muones 
    escalaEnergia1 = math.log(Energia[0],1.1)
    escalaEnergia2 = math.log(Energia[1],1.1)
    
    phiAxis1 = np.array([escalaEnergia1*4*np.pi/224])
    phiAxis2 = np.array([escalaEnergia2*4*np.pi/224])
    
    etaAxis1 = np.array([escalaEnergia1*12/224])
    etaAxis2 = np.array([escalaEnergia2*12/224])
    
    center1 = np.array([Eta[0],Phi[0]])
    center2 = np.array([Eta[1],Phi[1]])
    
    # Jets 1
    escalaEnergia3 = math.log(Energia[2],1.1)
    phiAxis3 = np.array([escalaEnergia3*4*np.pi/224])
    etaAxis3 = np.array([escalaEnergia3*12/224])
    center3 = np.array([Eta[2],Phi[2]])
    
    fig = plt.figure()
    fig.set_size_inches(4,4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis([-6,6,-2*np.pi,2*np.pi])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    Object1 = Ellipse(xy = center1, width=etaAxis1, height=phiAxis1, angle=0.0, facecolor = 'none', edgecolor= 'black', lw = 4)
    Object2 = Ellipse(xy = center2, width=etaAxis2, height=phiAxis2, angle=0.0, facecolor = 'none', edgecolor= 'black', lw = 4)
    Object3 = Ellipse(xy = center3, width=etaAxis3, height=phiAxis3, angle=0.0, facecolor = 'none', edgecolor= 'darkorange' , lw = 4)
    
    
    ax.add_artist(Object1)
    ax.add_artist(Object2)
    ax.add_artist(Object3)

    
    fig.savefig(ruta + str(cont)+ '.jpg', dpi=56)
    


# In[10]:


def images4(cont,Energia,Eta,Phi,ruta):
    
    # Muones 
    escalaEnergia1 = math.log(Energia[0],1.1)
    escalaEnergia2 = math.log(Energia[1],1.1)
    
    phiAxis1 = np.array([escalaEnergia1*4*np.pi/224])
    phiAxis2 = np.array([escalaEnergia2*4*np.pi/224])
    
    etaAxis1 = np.array([escalaEnergia1*12/224])
    etaAxis2 = np.array([escalaEnergia2*12/224])
    
    center1 = np.array([Eta[0],Phi[0]])
    center2 = np.array([Eta[1],Phi[1]])
    
    # Jets 1
    escalaEnergia3 = math.log(Energia[2],1.1)
    phiAxis3 = np.array([escalaEnergia3*4*np.pi/224])
    etaAxis3 = np.array([escalaEnergia3*12/224])
    center3 = np.array([Eta[2],Phi[2]])
    
    # Jets 2
    escalaEnergia4 = math.log(Energia[3],1.1)
    phiAxis4 = np.array([escalaEnergia4*4*np.pi/224])
    etaAxis4 = np.array([escalaEnergia4*12/224])
    center4 = np.array([Eta[3],Phi[3]])
    
    
    fig = plt.figure()
    fig.set_size_inches(4,4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis([-6,6,-2*np.pi,2*np.pi])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    Object1 = Ellipse(xy = center1, width=etaAxis1, height=phiAxis1, angle=0.0, facecolor = 'none', edgecolor= 'black', lw = 4)
    Object2 = Ellipse(xy = center2, width=etaAxis2, height=phiAxis2, angle=0.0, facecolor = 'none', edgecolor= 'black', lw = 4)
    Object3 = Ellipse(xy = center3, width=etaAxis3, height=phiAxis3, angle=0.0, facecolor = 'none', edgecolor= 'darkorange' , lw = 4)
    Object4 = Ellipse(xy = center4, width=etaAxis3, height=phiAxis3, angle=0.0, facecolor = 'none', edgecolor= 'darkorange' , lw = 4)
    
    
    ax.add_artist(Object1)
    ax.add_artist(Object2)
    ax.add_artist(Object3)
    ax.add_artist(Object4)

    
    fig.savefig(ruta + str(cont)+ '.jpg', dpi=56)
    


# In[34]:


# Dibuja los histogramas 

def Histogram(Data,name_dic,color):
    Histograma = {}
    Histograma[name_dic] = pd.DataFrame(Data)
    
    get_ipython().run_line_magic('matplotlib', 'inline')
    bins = np.linspace(0,200,201)
    Histograma[name_dic].plot.hist(bins=bins, alpha=0.5,histtype='stepfilled',color = color, figsize = (12,8))
    plt.xlabel('Masa invariante($ GeV/c^{2}$)',fontsize = 15)
    plt.ylabel('Frecuencia',fontsize = 15 )
    


# ## Análisis Higgs a dielectrones

# In[41]:


# Obtengo los datos del root
Prueba = Data('Datos/Roots/Higgs1_merged.root','events')


# In[37]:


Pt_H,Energia_H,Phi_H,Eta_H,cont = Z_to_Di_Lepton(Prueba,'muon')


# In[20]:


images(0,Energia_H[0][:],Eta_H[0][:],Phi_H[0][:],'Datos/Imagenes/H->ee/')


# In[21]:


cont = 0
for i in range(1,1193):
    images(cont+i,Energia_H[i][:],Eta_H[i][:],Phi_H[i][:],'Datos/Imagenes/H->ee/')


# In[38]:


masa_invariante_H = invariant_mass(Eta_H,Phi_H,Pt_H,Energia_H) 


# In[39]:


Histogram(masa_invariante_H,'masa_invariante_electron','darkorange')


# In[ ]:


#Numero de datos
# Higgs1_merged.root = 1193

