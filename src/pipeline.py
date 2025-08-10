import yaml

from pathlib import Path
from dataset import *
from model import *

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

NUM_CLASSES = 6

class WasteClassifier():
    def __init__(self, config_file: Path | str):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            self.model = WasteClassifierModule(config['lr'], config['model_size'], NUM_CLASSES)
            
            train_transform = transforms.Compose([
                pad_to_square, 
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p = config['horizontal_flip']),
                transforms.RandomVerticalFlip(p = config['vertical_flip']),
                transforms.RandomRotation(degrees=config['rotation']),
                transforms.ColorJitter(brightness=config['brightness'], contrast=config['contrast'], saturation=config['saturation'], hue=config['hue']),
                transforms.ToTensor(),
                transforms.Normalize([0.5581, 0.5410, 0.5185], [0.3177, 0.3070, 0.3034])
            ])
            
            val_test_transform = transforms.Compose([
                pad_to_square, 
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5581, 0.5410, 0.5185], [0.3177, 0.3070, 0.3034])
            ])
            
            self.data_module.setup(stage='fit')
            self.data_module.setup(stage='test')
            self.data_module.setup(stage='predict')
            
            self.data_module = WasteDatasetModule(config['dataset_path'], config['batch_size'], config['num_workers'], train_transform, val_test_transform)
            
            checkpoint = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_last=True,
                save_top_k=1,
                dirpath=config['checkpoint_dir'],
                filename=config['checkpoint_name'] + '_best',
                verbose=config['verbose']
            )
            
            checkpoint.CHECKPOINT_NAME_LAST = config['checkpoint_name'] + '_last'
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5)
            callbacks = [checkpoint, early_stop]

            tb_logger = TensorBoardLogger(
                save_dir=config['log_dir'],
                name=config['exp_name'],
                version=config['version']
            )
            
            csv_logger = CSVLogger(
                save_dir=config['metrics_dir'],
                name=config['exp_name'],
                version=config['version']
            )
            
            logger = [tb_logger, csv_logger]

            self.trainer = Trainer(
                max_epochs=config['epochs'],
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices= "auto",
                logger=logger,
                callbacks=callbacks,
            )
            
        except Exception as e:
            print(e)
            
    def train(self, last_checkpoint_path:str = None):
        self.data_module.setup('fit')
        self.trainer.fit(model=self.model, datamodule=self.data_module, ckpt_path=last_checkpoint_path)
    
    def test(self):
        self.data_module.setup('test')
        self.trainer.test(model=self.model, datamodule=self.data_module)
    
    def load_model(self, checkpoint_path: str):
        self.model = WasteClassifierModule.load_from_checkpoint(checkpoint_path)
    
    def predict(self):
        pass