from ultralytics import YOLO
import cv2
#PARA EL PROCESO DE IMAGENES
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from skimage.io import imread, imshow
from skimage.color import rgb2hsv


def Mask_red(img_cv2, hsv_img):
      lower_mask= hsv_img[:,:,0]<1
      upper_mask= hsv_img[:,:,0]>.9
      saturation_mask= hsv_img[:,:,1]>.5
      mask2= upper_mask*lower_mask*saturation_mask
      red= img_cv2[:,:,0]*mask2
      green= img_cv2[:,:,1]*mask2
      blue= img_cv2[:,:,2]*mask2
      img_masked =np.dstack((red,green,blue))
      return img_masked
# NUMERO 0 Rojo
def Mask_blue(img_cv2, hsv_img):
      lower_mask= hsv_img[:,:,0]<.62
      upper_mask= hsv_img[:,:,0]>.6
      saturation_mask= hsv_img[:,:,1]>.2
      mask2= upper_mask*lower_mask*saturation_mask
      red= img_cv2[:,:,0]*mask2
      green= img_cv2[:,:,1]*mask2
      blue= img_cv2[:,:,2]*mask2
      img_masked =np.dstack((red,green,blue))
      return img_masked
# Numero 1 AZUL
def Mask_green(img_cv2, hsv_img):
      lower_mask= hsv_img[:,:,0]<.56
      upper_mask= hsv_img[:,:,0]>.520
      saturation_mask= hsv_img[:,:,1]>.5
      mask2= upper_mask*lower_mask*saturation_mask
      red= img_cv2[:,:,0]*mask2
      green= img_cv2[:,:,1]*mask2
      blue= img_cv2[:,:,2]*mask2
      img_masked =np.dstack((red,green,blue))
      return img_masked
#Numero 2 VERDE
def Mask_yellow(img_cv2, hsv_img):
      lower_mask= hsv_img[:,:,0]<.130
      upper_mask= hsv_img[:,:,0]>.11
      saturation_mask= hsv_img[:,:,1]>.1
      mask2= upper_mask*lower_mask*saturation_mask
      red= img_cv2[:,:,0]*mask2
      green= img_cv2[:,:,1]*mask2
      blue= img_cv2[:,:,2]*mask2
      img_masked =np.dstack((red,green,blue))
      return img_masked  
#Numero 3 AMARILLO   
def Mask_orange(img_cv2, hsv_img):
      lower_mask= hsv_img[:,:,0]<.025
      upper_mask= hsv_img[:,:,0]>.0
      saturation_mask= hsv_img[:,:,1]>.3
      mask2= upper_mask*lower_mask*saturation_mask
      red= img_cv2[:,:,0]*mask2
      green= img_cv2[:,:,1]*mask2
      blue= img_cv2[:,:,2]*mask2
      img_masked =np.dstack((red,green,blue))
      return img_masked
#Numero 4 NARANJA
def Mask_purpule(img_cv2, hsv_img):
      lower_mask= hsv_img[:,:,0]<.665
      upper_mask= hsv_img[:,:,0]>.645
      saturation_mask= hsv_img[:,:,1]>.3
      mask2= upper_mask*lower_mask*saturation_mask
      red= img_cv2[:,:,0]*mask2
      green= img_cv2[:,:,1]*mask2
      blue= img_cv2[:,:,2]*mask2
      img_masked =np.dstack((red,green,blue))
      return img_masked
#Numero 5 MORADO

def Sharpening(img_masked):
      image = img_masked
      # Create shapening kernel, we don't normalize since the 
       # the values in the matrix sum to 1
      kernel_sharpening = np.array([[-1,-1,-1], 
                             [-1,9,-1], 
                             [-1,-1,-1]])
      sharpened = cv2.filter2D(image, -1, kernel_sharpening)
       # applying different kernels to the input image 
      return sharpened

def Opening_Closing( img_masked):
      image_op = img_masked
      image_cl = img_masked
      # Define the kernel size
      kernel = np.ones((5,5), np.uint8)
       # Opening - Good for removing noise
      opening = cv2.morphologyEx(image_op, cv2.MORPH_OPEN, kernel)
      #cv2.imshow('Opening', opening) # closing es mejor
      # Closing - Good for removing noise 
      closing = cv2.morphologyEx(image_cl, cv2.MORPH_CLOSE, kernel)
       # TErmina openinig an closing """
      return closing

def Dilatation_Erotion(img_masked):
      # Let's define our kernel size
      image = img_masked
      kernel = np.ones((5,5), np.uint8)
      
      # Now we erode
      erosion = cv2.erode(image, kernel, iterations = 1)
      
      dilation = cv2.dilate(image, kernel, iterations = 1)
      
      return dilation,erosion

def Adaptiv_Threshold( img_masked):
      
      image = img_masked
      grayscaled = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
      th = cv2.adaptiveThreshold(grayscaled, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,255,0)

      return th 
  
def Porcentaje( image, mask):
      pixeles = np.prod(image.shape[:2])
      color = np.sum(mask)
      return pixeles/color


def Percentage_1( th):
      eq = cv2.equalizeHist(th)
      hist = cv2.calcHist([th], [0], None, [256], [0, 256])
      total_pixels = th.size
      white_pixels = hist[255]+ hist[254] # Access the count of the highest intensity value
      white_percentage = (white_pixels / total_pixels) * 100
      
      return white_percentage   


model=YOLO('best_200_1.pt')


cap = cv2.VideoCapture(1)

row_low = 120 -100
row_up = 380 -100
col_low = 200 
col_up = 500 
counter = 10
A_counter = 0
PCA_counter =0
PRC_counter =0
PRM_counter =0
dectected_objs = []
detections = []
img_detected = 9
color_ant = 0
img_detected_ant = 0
estress = 0

while True:
    ret, frame = cap.read()
    img_cv2= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv_img = rgb2hsv(img_cv2)
    img_masked_R= Mask_red(img_cv2,hsv_img)
    img_masked_B= Mask_blue(img_cv2,hsv_img)
    img_masked_G= Mask_green(img_cv2,hsv_img)
    img_masked_Y= Mask_yellow(img_cv2,hsv_img)
    img_masked_O= Mask_orange(img_cv2,hsv_img)
    img_masked_P= Mask_purpule(img_cv2,hsv_img)

    th_R = Adaptiv_Threshold(img_masked_R)
    th_B= Adaptiv_Threshold(img_masked_B)
    th_G = Adaptiv_Threshold(img_masked_G)
    th_Y = Adaptiv_Threshold(img_masked_Y)
    th_O = Adaptiv_Threshold(img_masked_O)
    th_P = Adaptiv_Threshold(img_masked_P)

    percentage_R = Percentage_1(th_R)
    percentage_B = Percentage_1(th_B)
    percentage_G = Percentage_1(th_G)
    percentage_Y = Percentage_1(th_Y)
    percentage_O = Percentage_1(th_O)
    percentage_P = Percentage_1(th_P)

    array_cloth = (float(percentage_R), float(percentage_B), float(percentage_G),float(percentage_Y),float(percentage_O),float(percentage_P))

    last = 0
    pos = 0
    color = 0

    for item in array_cloth:
          
          if item > last:
                last = item
                color = pos
      
          pos += 1      

   

    recorte = frame[row_low:row_up, col_low:col_up]
    th = Adaptiv_Threshold(img_masked_Y)

    #cv2.imshow('img_masked_Y', img_cv2)


    
    results = model (recorte, show= True, conf=0.1, save=False)


    #YOLO EXTRACCION
    for result in results:
      for obj in result.boxes:
            class_id = int(obj.cls.item())
            conf = float(obj.conf.item())
            dectected_objs.append(class_id)
      if len(dectected_objs) != 0:
            detections.append(dectected_objs[0])
            dectected_objs =[]
            A_counter += 1
            if A_counter == 10:
                  counts = np.bincount(detections)
                  temp = np.argmax(counts)
                  img_detected = int(temp)
                  A_counter = 0
                  detections =[]
      elif len(dectected_objs) == 0:
            img_detected = 9          


    if img_detected == 0 or img_detected ==2:
          estress  =3 
    elif img_detected ==1 or img_detected ==6 or img_detected == 5:
          estress=2
    elif img_detected ==3 or img_detected == 4:
          estress =1


    if img_detected == img_detected_ant and color == color_ant:
          PCA_counter += 1
    else:
          PCA_counter = 0

    if   (color == 5 or color ==1 )and PCA_counter != 0:
          print("PCA")
          if PCA_counter == 30:
                if estress ==3:
                      print("Pulsaciones rapidas en Azul")
                elif estress ==2:
                      print("Pulsaciones lentas en Azul")
                elif estress ==1:
                      print("Iniciar pulsaciones rojas rapidas ")
          elif PCA_counter == 60:
                if estress ==3:
                      print("Pulsaciones medias en verde")
                elif estress ==2:
                      print("Transiciones pulsantes y lentas entre rojo verde y azul")
                elif estress ==1:
                      print("Pulsaciones medias en verde")
          elif PCA_counter ==90:
                if estress ==3:
                      print("Transiciones pulsantes y lentas entre rojo verde y azul")
                elif estress ==2:
                      print("Transiciones pulsantes y lentas entre rojo verde y azul")
                elif estress ==1:
                      print("Pulsaciones medias en verde")
          
    elif color == 3 and PCA_counter != 0:
          print ("PRM")
          if PCA_counter == 30:
                if estress ==3:
                      print("Pulsaciones rapidas en Azul")
                elif estress ==2:
                      print("Pulsaciones lentas en Azul")
                elif estress ==1:
                      print("Iniciar pulsaciones rojas rapidas ")
          elif PCA_counter == 60:
                if estress ==3:
                      print("Pulsaciones medias en rojo")
                elif estress ==2:
                      print("Transiciones pulsantes y lentas entre rojo verde y azul")
                elif estress ==1:
                      print("Pulsaciones medias en verde")
          elif PCA_counter ==90:
                if estress ==3:
                      print("Transiciones pulsantes y lentas entre rojo verde y azul")
                elif estress ==2:
                      print("Pulsaciones lentas en Azul")
                elif estress ==1:
                      print("Pulsaciones lentas en Azul")

    elif (color == 4 or color == 3 or color == 0) and PCA_counter != 0:
          print("PRC")
          if PCA_counter == 30:
                if estress ==3:
                      print("Pulsaciones rapidas en Azul")
                elif estress ==2:
                      print("Pulsaciones lentas en Azul")
                elif estress ==1:
                      print("Iniciar pulsaciones rojas rapidas ")
          elif PCA_counter == 60:
                if estress ==3:
                      print("Pulsaciones medias en verde")
                elif estress ==2:
                      print("Transiciones pulsantes y lentas entre rojo verde y azul")
                elif estress ==1:
                      print("Pulsaciones medias en verde")
          elif PCA_counter ==90:
                if estress ==3:
                      print("Transiciones pulsantes y lentas entre rojo verde y azul")
                elif estress ==2:
                      print("Transiciones pulsantes y lentas entre rojo verde y azul")
                elif estress ==1:
                      print("Pulsaciones medias en verde")
          
    elif color == 2 and PCA_counter != 0:
          print ("PRM")
          if PCA_counter == 30:
                if estress ==3:
                      print("Pulsaciones rapidas en Azul")
                elif estress ==2:
                      print("Pulsaciones lentas en Azul")
                elif estress ==1:
                      print("Iniciar pulsaciones rojas rapidas ")
          elif PCA_counter == 60:
                if estress ==3:
                      print("Pulsaciones medias en rojo")
                elif estress ==2:
                      print("Transiciones pulsantes y lentas entre rojo verde y azul")
                elif estress ==1:
                      print("Pulsaciones medias en verde")
          elif PCA_counter ==90:
                if estress ==3:
                      print("Transiciones pulsantes y lentas entre rojo verde y azul")
                elif estress ==2:
                      print("Pulsaciones lentas en Azul")
                elif estress ==1:
                      print("Pulsaciones lentas en Azul")



          
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(estress,color)
    print("counter",PCA_counter)
    img_detected_ant = img_detected
    color_ant =color

cap.release()
cv2.destroyAllWindows()
