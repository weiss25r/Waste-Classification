from pathlib import Path
from dataset import *
from model import *

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

NUM_CLASSES = 6

class WasteClassificationPipeline():
    def __init__(self, dataset_path: Path | str, lr, batch_size: int = 32):
        self.model = WasteClassifier(lr=lr, num_classes=NUM_CLASSES)
        self.data_module = WasteDatasetModule(dataset_path, batch_size=batch_size)
    
    def setup_train(self, epochs, exp_name, version, checkpoint_path=None):
        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_last=True,
            save_top_k=1,
            dirpath='../checkpoints/',
            filename='best',
            verbose=True
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        callbacks = [checkpoint, early_stop]

        logger = TensorBoardLogger(
            save_dir='../logs/',
            name=exp_name,
            version=version
        )

        self.trainer = Trainer(
            max_epochs=epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices= 1,
            logger=logger,
            callbacks=callbacks,
            
        )
        
        self.checkpoint_path = checkpoint_path
        self.setup = True
        
        
    def train(self):
        if self.setup == None:
            raise Exception('Please setup the pipeline before training using setup_train()')
        
        self.data_module.setup('fit')
        self.trainer.fit(model=self.model, datamodule=self.data_module, ckpt_path=self.checkpoint_path)
        
    def test(self):
        self.data_module.setup('test')
        self.model.eval()
        self.trainer.test(model=self.model, datamodule=self.data_module)