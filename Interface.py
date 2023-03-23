import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
import math

##################################################################
# FUNCIONES PARA RANDOM WALK
# Funcion para calculo de An
from numpy import cos,sin
def An(a,alfa,d,tetha):
    At = np.array([[cos(tetha), -sin(tetha)*cos(alfa), sin(tetha)*sin(alfa),a*cos(tetha)],
    [sin(tetha), cos(tetha)*cos(alfa),-cos(tetha)*sin(alfa),a*sin(tetha)],
    [0, sin(alfa), cos(alfa), d],
    [0,0,0,1]])
    
    return At

def CinemaDirect(punto):
  # Se pasa a radianes
  th1 = np.radians(punto[0])
  th2 = np.radians(punto[1])
  th3 = np.radians(punto[2])

  # Se declaran los parametros de denavit
  dhMtx = [
    [0,np.radians(90),9.5,th1],
    [10.5,0,0,th2],
    [7.5,0,0,th3]
  ]
  dhMtx = np.array(dhMtx)
  
  As = []
  for i in range(dhMtx.shape[0]):
      A = An(dhMtx[i,0],dhMtx[i,1],dhMtx[i,2],dhMtx[i,3])
      As.append(A)
  # As[0] es la matriz de transformacion 1
  
  Atotal = np.eye(4)
  for i in range(len(As)-1):
      Atotal = Atotal@As[i]
  punto2 = Atotal
  # Punto2 es la matriz de transformacion 2

  Atotal = np.eye(4)
  for i in range(len(As)):
      Atotal = Atotal@As[i]
  punto3 = Atotal
  # Punto3 es la matriz de transformacion 3

  return [As[0],punto2, punto3]

class RandWalk:
  def __init__(self, w = 0.08, actPos = [0,0,0]):
    self.w = w
    self.vecj = []
    self.jmin = 1
    self.actPos = actPos

  # Calculo de J entre dos puntos en el espacio de trabajo
  def jNomr(self,Pf,Pu):
    x = Pu[0]-Pf[0]
    y = Pu[1]-Pf[1]
    z = Pu[2]-Pf[2]
    dist = np.sqrt(x**2+y**2+z**2)
    return (dist)

  # Calculo de W
  def W(self,wb=0.05,nmax=0.1,phy=0.1):
    # Calculo de la derivada de j osea j punto (jp)
    if len(self.vecj) < 3:
      jp = 1000
    else:
      h1 = self.vecj[-1] - self.vecj[-2]
      h2 = self.vecj[-2] - self.vecj[-3]
      jp = h1-h2
    
    # Calculo de eta 
    n = nmax/(1+np.exp(-phy+jp))
    # Calculo de w
    self.w = wb*self.jmin + n

  # Cambia la posicion inicial 
  def NewPos(self, q1,q2,q3):
    self.actPos = [q1,q2,q3]

  def RandStep(self, pf, walkers = 30, alpha=5):
    pf = np.array(pf)
    # Condicion que el acercamiento es menor a alpha 
    # si no vuleve a calcular puntos
    mini = 10000
    while (alpha < mini): 

      nube = [] # Variable para la nube de puntos
      # Se calcula cierto numero de puntos (walkers)
      # alrededor de la posicion inicial (actPos) 
      for _ in range(walkers):
        step = np.random.uniform(low=-1, high=1, size=(1,3))
        step *= self.w

        # Condiciones de restriccion
        # ¿Es parte del conjunto de puntos prohibidos?


        nube.append(self.actPos + step)

      # Mide la distancia entre los puntos de la nube
      # y la posicion final deseada
      distancias = []
      for point in nube:
        MT = CinemaDirect(point[0]) # Matriz de transformacion
        Pxyz = MT[2][0:3,3]

        # Condiciones de restriccion
        x = Pxyz[0]
        y = Pxyz[1]
        z = Pxyz[2]

        # ¿Es alcanzable?
        if ((x)**2 + (y)**2 + (z - 9.5)**2 <= (18**2)): 
          distancias.append(self.jNomr(pf,Pxyz))
        else: 
          distancias.append(1000)


        #distancias.append(self.jNomr(pf,Pxyz))
      
      # ¿Quien se acerca mas?
      idx = np.argmin(distancias)
      mini = distancias[idx]
    
    # Aqui la distancia obtenida es menor a alpha
    # Se obtiene la j minima ultima
    self.jmin = distancias[idx]
    # Se guarda la ultima distancia j minima
    self.vecj.append(self.jmin)
    # Actualiza w para una futura iteracion
    self.W()
    # Se actualiza la posicion actual 
    self.actPos = nube[idx][0]

    return (self.actPos)

  def WalkTo(self, PxyzFin, limSteps = 1000, finalStep = 0.1):
    # PxyzFinal = Punto final en coordenadas de trabajo
    # limSteps = Limite de pasos por si se queda trabado
    # finalStep = Radio del ultimo paso para que se acerque 

    self.steps = [] # Pasos para llegar a la posicion final
    self.steps.append(self.actPos)

    # Calulamos la maxima distancia entre el punto actual y el deseado
    MT = CinemaDirect(self.actPos) # Matriz de transformacion
    Pxyz = MT[2][0:3,3]
    actualMaxDist = self.jNomr(PxyzFin,Pxyz)

    while ((len(self.steps) <= limSteps+1 ) and (self.jmin >= finalStep )):
      pos = self.RandStep(PxyzFin, alpha = actualMaxDist) 
      self.steps.append(pos)
    
    return(self.steps)


##################################################################
# INTERFAZ 

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Crear los contenedores
        hbox = QHBoxLayout()
        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()

        # Agregar el gráfico al primer contenedor
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Definimos los puntos iniciales en la posicion actual asi como los finales por default
        self.actPoints = [[0,0,9.5,],
                    [10.5,0,9.5],
                    [18,0,9.5]]
        self.finalPoints = self.actPoints
        
        # Definimos los puntos
        puntos = np.array([[0,0,0],
                    self.actPoints[0],
                    self.actPoints[1],
                    self.actPoints[2]])

        # Graficamos los puntos
        self.ax.scatter(puntos[:,0], puntos[:,1], puntos[:,2],c='r', marker='o')

        # Unimos los puntos con líneas
        self.ax.plot(puntos[:,0], puntos[:,1], puntos[:,2])

        # Fijar los límites de los ejes
        self.ax.set_xlim([-18,18])
        self.ax.set_ylim([-18, 18])
        self.ax.set_zlim([0, 28])

        self.canvas = FigureCanvas(self.fig)
        

        # Agregar widgets al primer contenedor
        label1 = QLabel("Simulacion")
        label1_font = QFont()
        label1_font.setBold(True)
        label1.setFont(label1_font)
        vbox1.addWidget(label1, 0, alignment=Qt.AlignCenter)
        vbox1.addWidget(self.canvas)

        # Agregar widgets al segundo contenedor
        label2 = QLabel("Panel de control")
        label2_font = QFont()
        label2_font.setBold(True)
        label2.setFont(label2_font)
        vbox2.addWidget(label2, 0, alignment=Qt.AlignCenter)

            # Items
        posLabel1 = QLabel("Posicion inicial")
        posLabel2 = QLabel("Posicion final")
        self.ap1 = QLabel("Eslabon 1: " + str(self.actPoints[0]))
        fp1 = QLabel("Eslabon 1: " + str(self.finalPoints[0]))
        self.ap2 = QLabel("Eslabon 2: " + str(self.actPoints[1]))
        fp2 = QLabel("Eslabon 2: " + str(self.finalPoints[1]))
        self.ap3 = QLabel("Eslabon 3: " + str(self.actPoints[2]))
        self.fp3 = QLabel("Eslabon 3: " + str(self.finalPoints[2]))

        Xlabel = QLabel("X")
        Ylabel = QLabel("Y")
        Zlabel = QLabel("Z")
        boton1 = QPushButton("+")
        boton1.clicked.connect(self.btn1)
        boton2 = QPushButton("-")
        boton2.clicked.connect(self.btn2)
        boton3 = QPushButton("+")
        boton3.clicked.connect(self.btn3)
        boton4 = QPushButton("-")
        boton4.clicked.connect(self.btn4)
        boton5 = QPushButton("+")
        boton5.clicked.connect(self.btn5)
        boton6 = QPushButton("-")
        boton6.clicked.connect(self.btn6)

        boton7 = QPushButton("Correr simulacion")
        boton7.clicked.connect(self.btn7)
        boton8 = QPushButton("Ejecutar movimiento")
        boton8.clicked.connect(self.btn8)

        # Botones en la caja X
        x_box = QHBoxLayout()
        x_box.addWidget(boton1)
        x_box.addWidget(Xlabel, 0, alignment=Qt.AlignCenter)
        x_box.addWidget(boton2)
        
        # Botones en la caja Y
        y_box = QHBoxLayout()
        y_box.addWidget(boton3)
        y_box.addWidget(Ylabel, 0, alignment=Qt.AlignCenter)
        y_box.addWidget(boton4)

        # Botones en la caja Z
        z_box = QHBoxLayout()
        z_box.addWidget(boton5)
        z_box.addWidget(Zlabel, 0, alignment=Qt.AlignCenter)
        z_box.addWidget(boton6)

        # Agregar elementos al contenedor 2
        vbox2.addWidget(posLabel1)
        vbox2.addWidget(self.ap1, 0, alignment=Qt.AlignCenter)
        vbox2.addWidget(self.ap2, 0, alignment=Qt.AlignCenter)
        vbox2.addWidget(self.ap3, 0, alignment=Qt.AlignCenter)

        vbox2.addWidget(posLabel2)
        #vbox2.addWidget(fp1, 0, alignment=Qt.AlignCenter)
        #vbox2.addWidget(fp2, 0, alignment=Qt.AlignCenter)
        vbox2.addWidget(self.fp3, 0, alignment=Qt.AlignCenter)

            # Añadir cajas de botones al contenedor 2
        vbox2.addLayout(x_box)
        vbox2.addLayout(y_box)
        vbox2.addLayout(z_box)

        vbox2.addWidget(boton7)
        vbox2.addWidget(boton8)

        # Agregar los contenedores al contenedor horizontal
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)

        # Agregar el contenedor horizontal a la ventana principal
        self.setLayout(hbox)

        # Configurar la ventana principal
        self.setWindowTitle("NeuroBØT controller ")
        self.setGeometry(100, 100, 800, 400)

        # Agregar un ícono a la ventana principal
        icon = QIcon("Icon.png")
        self.setWindowIcon(icon)

    # Funciones de los botones
    def btn1(self):
        self.finalPoints[2][0] += 0.1
        self.fp3.setText("Eslabon 3: " + str(self.finalPoints[2]))
    
    def btn2(self):
        self.finalPoints[2][0] -= 0.1
        self.fp3.setText("Eslabon 3: " + str(self.finalPoints[2]))

    def btn3(self):
        self.finalPoints[2][1] += 0.1
        self.fp3.setText("Eslabon 3: " + str(self.finalPoints[2]))

    def btn4(self):
        self.finalPoints[2][1] -= 0.1
        self.fp3.setText("Eslabon 3: " + str(self.finalPoints[2]))

    def btn5(self):
        self.finalPoints[2][2] += 0.1
        self.fp3.setText("Eslabon 3: " + str(self.finalPoints[2]))

    def btn6(self):
        self.finalPoints[2][2] -= 0.1
        self.fp3.setText("Eslabon 3: " + str(self.finalPoints[2]))

    def btn7(self):
        Walker = RandWalk()
        punto = self.finalPoints[2]
        # print(punto)
        ruta = Walker.WalkTo(punto)
        print('Pasos en ruta: ')
        print(len(ruta))
        # print('Espacio articular: ')
        # print(ruta[-1])
        R = CinemaDirect(ruta[-1])
        # print('Posicion del ultimo eslabon: ')

        # Borramos el grafico anterior
        self.ax.cla()
        
        # Definimos los puntos
        puntos = np.array([[0,0,0],
                          R[0][0:3,3],
                          R[1][0:3,3],
                          R[2][0:3,3]])
        
        # Actualizamos la posicion actual 
        self.actPoints =np.array([R[0][0:3,3],
                          R[1][0:3,3],
                          R[2][0:3,3]])
        self.ap1.setText("Eslabon 1: " + str(self.actPoints[0]))
        self.ap2.setText("Eslabon 2: " + str(self.actPoints[1]))
        self.ap3.setText("Eslabon 3: " + str(self.actPoints[2]))
        
        # Graficamos los puntos
        self.ax.scatter(puntos[:,0], puntos[:,1], puntos[:,2],c='r', marker='o')

        # Unimos los puntos con líneas
        self.ax.plot(puntos[:,0], puntos[:,1], puntos[:,2])

        # Fijar los límites de los ejes
        self.ax.set_xlim([-18,18])
        self.ax.set_ylim([-18, 18])
        self.ax.set_zlim([0, 28])

        #Actualizar grafico
        self.canvas.draw()

        for step in ruta:
          print('Espacio articular: ')
          print(step)
          R = CinemaDirect(step)
          coor = np.array([R[0][0:3,3],
                          R[1][0:3,3],
                          R[2][0:3,3]])
          print('Coordenadas trabajo: ')
          print(coor)
          

    def btn8(self):
        print("Botón 6 fue presionado")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
