# Guía de publicación en GitHub

## 1. Preparar el repositorio

1. Copiar `segmentation_bundle.joblib` desde la aplicación local a `artifacts/`.
2. Sustituir `TU_USUARIO` en `CITATION.cff` y `ANEXO_B_TEXTO.txt`.
3. Confirmar que no existen `.venv`, `kaggle.json`, `.env`, datos brutos ni archivos temporales.

## 2. Crear el repositorio con GitHub Desktop

1. Instalar GitHub Desktop e iniciar sesión.
2. Crear un repositorio nuevo llamado `tfm-segmentacion-clientes-bancarios`.
3. Copiar todo el contenido de esta plantilla dentro de la carpeta local creada por GitHub Desktop.
4. En GitHub Desktop, escribir el resumen del commit: `Publicación inicial del TFM`.
5. Pulsar `Commit to main`.
6. Pulsar `Publish repository` y desmarcar `Keep this code private` para publicarlo.

## 3. Crear la release

1. Abrir el repositorio en GitHub.
2. Entrar en `Releases` y seleccionar `Draft a new release`.
3. Crear la etiqueta `v1.0.0-tfm`.
4. Usar el título `Versión asociada al depósito del TFM`.
5. Copiar el contenido de `RELEASE_NOTES_v1.0.0.md`.
6. Publicar la release.

## 4. Incorporar el enlace al TFM

Sustituir `TU_USUARIO` en `ANEXO_B_TEXTO.txt`, copiar el texto al documento principal y actualizar el índice.
