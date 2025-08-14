import yaml

from pathlib import Path
from dataset import *
from model import *

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

NUM_CLASSES = 6

class WasteClassifierTrainer():
    def __init__(self, config_file: Path | str):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            self.model = WasteClassifierModule(config['lr'], config['model_size'], NUM_CLASSES)
            
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(p = config['horizontal_flip']),
                transforms.RandomVerticalFlip(p = config['vertical_flip']),
                transforms.RandomRotation(degrees=config['rotation']),
                transforms.ColorJitter(brightness=config['brightness'], contrast=config['contrast'], saturation=config['saturation'], hue=config['hue']),
                transforms.ToTensor(),
                transforms.Normalize([0.5581, 0.5410, 0.5185], [0.3177, 0.3070, 0.3034])
            ])
            
            self.eval_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5581, 0.5410, 0.5185], [0.3177, 0.3070, 0.3034])
            ])
            
            self.data_module = WasteDatasetModule(config['dataset_path'], config['batch_size'], config['num_workers'], train_transform, self.eval_transform)
            
            checkpoint = ModelCheckpoint(
                monitor='val_MulticlassF1Score',
                mode='max',
                save_last=True,
                save_top_k=1,
                dirpath=config['checkpoint_dir'],
                filename=config['checkpoint_name'] + '_best',
                verbose=config['verbose']
            )
            
            checkpoint.CHECKPOINT_NAME_LAST = config['checkpoint_name'] + '_last'
            
            early_stop = EarlyStopping(monitor='val_loss', patience=config['patience'])
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
    
    def predict_on_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        
        img_ready = self.eval_transform(img)
        img_batch = img_ready.unsqueeze(0)
        
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(img_batch)
            pred_idx = torch.argmax(logits, dim=1).item()
            
        class_name = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
        predicted_class = class_name[pred_idx]
        
        return predicted_class