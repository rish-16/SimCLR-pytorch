# SimCLR-pytorch
Unofficial PyTorch implementation of SimCLR by Google Brain [[`abs`](https://arxiv.org/abs/2002.05709), [`pdf`](https://arxiv.org/pdf/2002.05709.pdf)]

## Installation
You can install the package via `pip`:

```bash
pip install simclr_pytorch
```

## Usage
You can use the `SimCLR` model like so:

```python
from simclr_pytorch import SimCLR

train_loader = ...
bigcnn = ResNet50(...)
simclr = SimCLR(
            image_size=(32, 32),
            bigcnn=bigcnn
        )

trained_bigcnn = simclr.fit(train_loader)
```

## Contributions
If you have any suggestions or hiccups, feel free to raise a PR or Issue. All contributions welcome!

## License
[MIT](https://github.com/rish-16/SimCLR-pytorch/blob/main/LICENSE)