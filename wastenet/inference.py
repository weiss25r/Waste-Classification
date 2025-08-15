from torchvision.transforms.functional import resize, center_crop, normalize, to_tensor
import onnxruntime
import numpy as np
from PIL import Image

class InferenceSession():
    def __init__(self, onnx_model_path: str):
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)

    def make_prediction(self, img: Image.Image):
        img = resize(img, 256)
        img = center_crop(img, 224)
        img = to_tensor(img)
        img = normalize(img, [0.5581, 0.5410, 0.5185], [0.3177, 0.3070, 0.3034])
        img = img.unsqueeze(0)
        
        input_name = self.ort_session.get_inputs()[0].name
        output = self.ort_session.run(None, {input_name: img.numpy()})

        classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
        predicted_class = classes[np.argmax(output)]

        return predicted_class