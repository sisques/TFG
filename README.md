# DYnamicCNN
Dinamic object masking CNN using Yolact

Dependencias:

Necesario tener instalado Anaconda3, cmake, make y git. 

Se recomienda la ejecución en un entorno Linux

Instalación:

1. Ejecutar `setup.py` para descargar e instalar las dependencias

2 Descargar los pesos de YOLACT y meterlos en la carpeta weights/yolact

| Weights                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------|
| [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)  | 
| [yolact_darknet53_54_800000.pth](https://drive.google.com/file/d/1dukLrTzZQEuhzitGkHaGjphlmRJOjVnP/view?usp=sharing) | 
| [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing)      | 
| [yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view?usp=sharing)     | 

3. Descargar los pesos de DYnamicCNN y meterlos en la carpeta weights/DYnamicCNN

| Weights                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------|
| [reg_medium_50_5e-5.h5] (https://drive.google.com/file/d/1-1JKGBxwXdwqDW1zwZm00Eg5sYF9DBB_/view?usp=sharing)  |


Detección:

1. Ejecutar el fichero `detect.py`, especificando 2 imagenes de entrada y una ruta de salida de la siguiente manera:

`python detect.py --inputImage1=PATH/TO/IMAGE --inputImage2=PATH/TO/IMAGE --outputPath=OUTPUT/PATH'

