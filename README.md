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

## TODO

- [ ] Complete necessary methods for `DpicNet()`
- [ ] Create `main.py` under `src/` for executing DpicNet
- [ ] Train/test on newly added FC layers (changing layers, nodes, biases, regularization, etc.)
- [ ] Predict and put result into report
- [ ] Complete report located under `report/`