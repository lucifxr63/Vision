# Alzheimers (Baseline CXR con DenseNet-121)

Este proyecto reproduce el baseline visto: un clasificador de **radiografías de tórax** con
**transfer learning** usando **DenseNet121** (estilo *CheXNet*) para **tres clases** (COVID / No-COVID / Normal).

## Estructura esperada de datos
Coloca las carpetas en `./data` (puedes cambiar rutas por CLI):


data/
Train/
COVID/
NonCOVID/
Normal/
Val/
COVID/
NonCOVID/
Normal/
Test/
COVID/
NonCOVID/
Normal/


## Requisitos
- Python 3.9+
- TensorFlow/Keras 2.x

Instalación rápida:
```bash
bash run.sh

Uso

Entrenamiento + evaluación (con curvas y matriz de confusión):

python train_eval.py \
  --train_dir ./data/Train \
  --val_dir ./data/Val \
  --test_dir ./data/Test \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3


(Optativo) Si tienes pesos CheXNet (.h5):

python train_eval.py ... --chexnet_weights ./weights/brucechou1983_CheXNet_Keras_0.3.0_weights.h5


Artefactos:

./models/best.h5 (mejor modelo por val_accuracy)

./artifacts/train_curve.png, ./artifacts/Cm_test.png

./artifacts/classification_report_test.txt

Referencias

Huang et al., Densely Connected Convolutional Networks, CVPR 2017.

Rajpurkar et al., CheXNet (DenseNet121 aplicado a CXR), arXiv 2017.


