import torch
import torchvision

from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from lightning import LightningModule
from torch import nn

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, PrecisionRecallCurve, ConfusionMatrix
from lightning.pytorch.loggers import TensorBoardLogger


class WasteClassifierModule(LightningModule):
    
    def __init__(self, lr=0.001, model_size: str = 'large', num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        
        self.save_hyperparameters()
        
        if model_size == 'large':
            self.model = mobilenet_v3_large(weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
        elif model_size == 'small':
            self.model = mobilenet_v3_small(weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
        else:
            raise ValueError(f"Please choose the model size between large and small.")
        
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes), MulticlassPrecision(num_classes, average='weighted'), 
            MulticlassRecall(num_classes, average='weighted'), MulticlassF1Score(num_classes, average='weighted')
        ])
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        
        self.pr_curve_micro = PrecisionRecallCurve(task='multiclass', num_classes=num_classes, average='micro')
        self.pr_curve = PrecisionRecallCurve(task='multiclass', num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=False, prog_bar=True, on_epoch=True )
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
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        
        preds = torch.argmax(logits, dim=1)
        self.valid_metrics.update(logits, y)
        
    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute())
        self.valid_metrics.reset()
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        
        self.pr_curve.update(logits, y)
        self.pr_curve_micro.update(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_metrics.update(preds, y)
        self.confusion_matrix.update(preds, y)
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                self.tensorboard = logger
                
        fig, ax = self.pr_curve_micro.plot()
        ax.set_title("PR Curve (Micro)")
        
        fig.canvas.draw()
        self.tensorboard.experiment.add_figure("PR Curve (Micro)", fig, global_step=self.global_step)
        
        fig, ax = self.pr_curve.plot()
        ax.set_title("PR Curve (All classes)")
        ax.legend(['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash'])
        fig.canvas.draw()
        self.tensorboard.experiment.add_figure("PR Curve (All classes)", fig, global_step=self.global_step)
        
        fig, ax = self.confusion_matrix.plot()
        ax.set_title("Confusion Matrix")
        fig.canvas.draw()
        self.tensorboard.experiment.add_figure("Confusion Matrix", fig, global_step=self.global_step)
        
        self.test_metrics.reset()
        self.pr_curve.reset()
        self.pr_curve_micro.reset()
        self.confusion_matrix.reset()
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds, y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)