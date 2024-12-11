from ultralytics import YOLO
import yaml

def train_model(params):
    model_arch = params["train"]["arch"]
    epochs = params["train"]["fine_tune_args"]["epochs"]
    img_size = params["train"]["img_size"]
    
    model = YOLO(model_arch)
    model.train(data="C:/Users/teachbricks/Desktop/MLops/data.yaml", epochs=epochs, imgsz=img_size)

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train_model(params)
