import random
import time
import base64
import json
import cv2
import numpy as np
import torch
from torchvision.models import detection
from paho.mqtt import client as mqtt_client


def webcam_reader(dev):
    def lst_to_batch(lst):
        out_reso = (427, 240)
        lst = [cv2.resize(img, out_reso, interpolation=cv2.INTER_LINEAR)
               for img in lst]
        batch = np.array(lst)
        batch = batch[:, :, ::-1] # bgr to rgb
        batch = np.transpose(batch, (0, 3, 1, 2)) # HWC to CHW
        batch = np.ascontiguousarray(batch)
        batch = torch.tensor(batch, device=dev, dtype=torch.float32)
        batch /= 255
        return batch, lst

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    frame_id = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        yield lst_to_batch([img])
        frame_id += 1
    cap.release()


class Publisher:
    def __init__(self, broker='localhost', port=1883):
        self.broker = broker
        self.port = port
        self.topic = "webcam-detections"
        self.client_id = f'python-mqtt-{random.randint(0, 1000)}'
        # username = 'dmitrii'
        # password = 'public'
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = self.model.to(self.dev)
        self.model.eval()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    def connect_mqtt(self):
        client = mqtt_client.Client(self.client_id)
        # client.username_pw_set(username, password)
        client.on_connect = self.on_connect
        client.connect(self.broker, self.port)
        return client

    def publish(self, client):
        msg_count = 0
        reader = webcam_reader(self.dev)
        while True:
            try:
                batch, img_list = next(reader)
                with torch.no_grad():
                    pred = self.model(batch)[0]
                boxes = pred['boxes']
                labels = pred['labels']
                scores = pred['scores']
                # filter persons with the score of at least 0.9
                person_label = 1
                min_score = 0.9
                shortlist = [box for box, label, score in zip(boxes, labels, scores)
                             if label == person_label and score >= min_score]
                shortlist = [b.cpu().numpy() for b in shortlist]
                # find the biggest person in terms of its pixel area
                if len(shortlist) > 0:
                    biggest = max(shortlist, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
                    biggest = biggest.tolist()
                else:
                    biggest = None
                img_base64 = base64.b64encode(img_list[0]).decode("utf-8") 
                message_dict = dict(frame_id=msg_count,
                                    image_base64=img_base64,
                                    shape=list(img_list[0].shape),
                                    biggest=biggest)
                msg = json.dumps(message_dict)
                result = client.publish(self.topic, msg)
                status = result[0]
                if status == 0:
                    print(f"Send `{list(message_dict.keys())}` to topic `{self.topic}`")
                else:
                    print(f"Failed to send message to topic {self.topic}")
                msg_count += 1
            except KeyboardInterrupt as ex:
                print("Exiting")
                break
            except StopIteration as ex:
                print("Exiting")
                break
        del reader

    def run(self):
        client = self.connect_mqtt()
        client.loop_start()
        self.publish(client)


if __name__ == '__main__':
    publisher = Publisher()
    publisher.run()
