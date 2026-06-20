# Artefacto del modelo

Antes de publicar el repositorio, copie en esta carpeta el archivo generado por el notebook:

```text
segmentation_bundle.joblib
```

Ruta esperada por la aplicación:

```text
artifacts/segmentation_bundle.joblib
```

El archivo contiene el modelo K-Means final, el escalador robusto, los límites de winsorización, el orden de variables, las transformaciones seleccionadas, los umbrales de elegibilidad y el playbook de segmentos.

No cargue artefactos obtenidos de fuentes no confiables. Los archivos persistidos con `joblib` deben utilizarse únicamente desde el repositorio y entorno controlados del proyecto.
