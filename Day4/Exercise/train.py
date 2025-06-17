from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    model.train(data = r"D:\桌面\shixun\ultralytics-main\TrafficSignDetection.v9\TrafficSignDetection.v9.yaml",imgsz=640,epochs=20,batch=16)

