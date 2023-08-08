## Install
```bash
pip install ultr-toolbox
```

## Click Models

### Create Datasets
```Python
from ultr_toolbox.click_models.data import ClickDataset

train_dataset = ClickDataset(train_df)
val_dataset = ClickDataset(val_df)
test_dataset = ClickDataset(test_df)
```

### Train neural click models
```
from ultr_toolbox.click_models.metrics import Perplexity
from ultr_toolbox.click_models.neural import PositionBasedModel, NeuralTrainer

model = PositionBasedModel()
trainer = NeuralTrainer(model)
trainer.fit(train_dataset, val_dataset)
metrics = trainer.test(test_dataset, metrics=[Perplexity()])
```

### Train PyClick models
To optionally train click models from the [PyClick](https://github.com/markovi/PyClick) library,
first install PyClick as a dependency:
```bash
pip install git+https://github.com/markovi/PyClick
```

Next, you can use the `PyClickTrainer` module to run the same pipeline as for the Jax-based neural click models:
```
from pyclick.click_models import PBM

from ultr_toolbox.click_models.metrics import Perplexity
from ultr_toolbox.click_models.em import PyClickTrainer

model = PBM()
trainer = PyClickTrainer(model)
trainer.fit(train_dataset, val_dataset)
metrics = trainer.test(test_dataset, metrics=[Perplexity()])
```

### Train naive models based on click statistics
```
from ultr_toolbox.click_models.metrics import Perplexity
from ultr_toolbox.click_models.stats import StatsTrainer, RankDocumentBasedModel

model = RankDocumentBasedModel()
trainer = StatsTrainer(model)
trainer.fit(train_dataset, val_dataset)
metrics = trainer.test(test_dataset, metrics=[Perplexity()])
```
