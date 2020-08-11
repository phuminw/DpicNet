# DpicNet

Deep Neural Network tailored-made for [Intel Multi-class Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification).

## Dataset

The dataset of DpicNet comes from [Intel Multi-class Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification), which is located in the following structure

```
src/
    data/
        predict/
        test/
            buildings/
            forest/
            glacier/
            mountain/
            sea/
            street/
        train/
            buildings/
            forest/
            glacier/
            mountain/
            sea/
            street/
```

Training and testing data are loaded by functions located in [data.py](src/model/data.py) with associated classes being inferred from the directory structure.

See [report.pdf](report/report.pdf) for more details
