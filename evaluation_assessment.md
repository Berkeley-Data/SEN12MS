1. The difference evaluation options:
> - a. Land cover classification -- image classificaiton.
> - b. Semantic Segmentation -- assigning a class label to every pixel of the input image.

2. 

3. tmp
> a. SEN12MS - Supervised Learning Benchmark - Classification

| Backbone  | Land Type  | Modalitities  | Bactch size  | Epochs | Accuracy (%) | Macro-F1 (%) | Micro-F1 (%) |
|---|---|---|---|---|---|---|---|
|DenseNet|single-label|_s1s2|64|100|51.16|50.78|62.90|
|DenseNet|single-label|_s2|64|100|54.41|52.32|64.74|
|ResNet50|single-label|_RGB|64|100|45.11|45.16|58.98|
|ResNet50|single-label|_s1s2|64|100|45.52|53.21|64.66|
|ResNet50|single-label|_s2|64|100|57.33|53.39|66.35|
|ResNet50|multi-label|_RGB|64|100|89.86|47.57|66.51|
|ResNet50|multi-label|_s1s2|64|100|91.22|57.46|71.40|
|ResNet50|multi-label|_s2|64|100|90.62|56.14|69.88|