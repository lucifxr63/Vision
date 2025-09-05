📄 Propuesta de Proyecto
i. Definición de proyecto: Importancia y contexto

La pandemia de COVID-19 puso en evidencia la necesidad de herramientas automáticas de diagnóstico rápido y preciso. Aunque las pruebas PCR siguen siendo el estándar, presentan limitaciones de costo, tiempo y disponibilidad en ciertos contextos.
El uso de imágenes médicas, en particular radiografías de tórax (CXR), constituye una alternativa rápida, económica y accesible, especialmente en hospitales con infraestructura limitada.

El presente proyecto busca desarrollar un sistema de clasificación y segmentación de imágenes CXR capaz de:

Distinguir entre pacientes con COVID-19, infecciones no-COVID (bacterianas o virales) y normales.

Usar máscaras de segmentación de pulmones e infecciones para mejorar la precisión y la interpretabilidad.

Explorar la innovación MoE (Mixture of Experts) + Test-Time Augmentation (TTA), con el objetivo de aumentar la robustez y generalización del modelo en la etapa de inferencia.

La importancia de este proyecto radica en su potencial apoyo clínico para médicos, permitiendo diagnósticos más rápidos y seguimiento de la progresión de la enfermedad.

ii. Descripción de datos

El dataset seleccionado es COVID-QU-Ex
, publicado en Kaggle y desarrollado por investigadores de Qatar University.

Fuente de datos: Kaggle (Anas Mohammed Tahir et al., 2021).

Cantidad de imágenes:

Total: 33,920 CXR

COVID-19: 11,956

No-COVID (pneumonías virales/bacterianas): 11,263

Normal: 10,701

Máscaras de segmentación:

Pulmones (ground-truth disponibles para todas las imágenes).

Infecciones (subconjunto de 5,826 imágenes: 2,913 COVID-19, 1,457 No-COVID y 1,456 normales).

Número de clases: 3 (COVID-19, No-COVID, Normal).

Resolución: Varía según la fuente original, pero estandarizadas en el dataset (~512×512 píxeles).

Objetos en imágenes: Cada imagen contiene la región pulmonar segmentada, con posibilidad de anotar infecciones.

iii. Diseño de experimentos

Se plantean los siguientes experimentos:

Preprocesamiento y segmentación:

Uso de las máscaras de pulmones para recortar región de interés.

Normalización y estandarización de resoluciones.

Clasificación básica (baseline):

CNN estándar (ej. ResNet-50, EfficientNet).

Entrenamiento con división hold-out:

70% training, 15% validación, 15% test.

Evaluación inicial sin innovaciones.

Implementación de MoE (Mixture of Experts):

Arquitectura que combina múltiples clasificadores especializados en diferentes regiones/características.

Los expertos se entrenan en subsets o features, y un "gating network" decide la ponderación de cada uno.

Test-Time Augmentation (TTA):

Durante la inferencia se aplican transformaciones (rotaciones, flips horizontales, cambios de contraste).

El resultado final se obtiene promediando las predicciones de las versiones augmentadas.

MoE + TTA (Innovación):

Integración de ambos enfoques para mejorar generalización.

Hipótesis: la combinación aumentará la robustez contra ruido y reducirá overfitting.

iv. Evaluación

Se utilizarán al menos dos métricas principales, además de otras secundarias:

Accuracy (Exactitud): proporción de clasificaciones correctas.

F1-Score macro: balance entre precisión y recall para todas las clases, ideal en datasets con clases ligeramente desbalanceadas.

(Opcional) AUC-ROC por clase: para analizar la capacidad discriminativa entre COVID vs. No-COVID.

(Opcional) IoU (Intersection over Union): en caso de evaluar segmentación de infecciones.