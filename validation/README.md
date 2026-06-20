# Evidencias de validación

Esta carpeta contiene resultados resumidos de la validación fuera de muestra:

- `validacion_holdout_resumen.csv`: estadísticos agregados de ARI, NMI, concordancia alineada y distancia al centroide.
- `validacion_holdout_repeticiones.csv`: resultados de las 20 particiones estratificadas de entrenamiento/holdout.

La partición final del TFM se utiliza como referencia estructural. Las métricas no representan exactitud supervisada, porque el clustering no dispone de etiquetas verdaderas externas.

Las pruebas funcionales adicionales —cliente elegible, historial insuficiente, lote mixto y recuperación de los cuatro segmentos— pueden reproducirse con el notebook y los CSV sintéticos de la carpeta `data/`.
