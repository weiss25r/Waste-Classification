import torch
import torchvision
from torchvision.models import mobilenet_v3_small
from lightning import LightningModule
from torch import nn

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

class WasteClassifier(LightningModule):
    
    def __init__(self, lr=0.001, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        
        self.save_hyperparameters()
        self.model = mobilenet_v3_small(weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, 6)
        
        self.criterion = nn.CrossEntropyLoss()
        
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes), MulticlassPrecision(num_classes), 
            MulticlassRecall(num_classes), MulticlassF1Score(num_classes)
        ])
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_epoch=True )
        preds = torch.argmax(logits, dim=1)
        self.train_metrics.update(preds, y)
        
        return loss
    
    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        preds = torch.argmax(logits, dim=1)
        self.valid_metrics.update(preds, y)
        
    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute())
        self.valid_metrics.reset()
            
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        
        preds = torch.argmax(logits, dim=1)
        
        self.test_metrics.update(preds, y)
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds, y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
