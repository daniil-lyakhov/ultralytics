from ultralytics import YOLO
import nncf
import torch

# Load a model
ckpt_path = "yolov8n.pt"
ckpt_path="/home/dlyakhov/Projects/ultralytics/runs/detect/train35/weights/best.pt"
model = YOLO(ckpt_path)  # load a pretrained model (recommended for training)

#quantized_model = nncf.quantize(model.model, nncf.Dataset([torch.zeros(1, 3,320,640)]*10,))



# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format