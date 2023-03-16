import random
import time
from threading import Thread, current_thread
from queue import Queue
from paho.mqtt import client as mqtt_client
import http.server
import socketserver
import os
import numpy as np
import cv2
import json
import base64


class Subscriber:
    def __init__(self, message_queue, broker='localhost', port=1883):
        self.message_queue = message_queue
        self.broker = broker
        self.port = port
        self.topic = "webcam-detections"
        self.client_id = f'python-mqtt-{random.randint(0, 1000)}'

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    def connect_mqtt(self):
        client = mqtt_client.Client(self.client_id)
        client.on_connect = self.on_connect
        client.connect(self.broker, self.port)
        return client

    def on_message(self, client, userdata, msg):
        decoded = msg.payload.decode()
        print(f"Received `{len(decoded)}` bytes from `{msg.topic}` topic")
        try:
            self.message_queue.put(decoded, block=False)
        except:
            pass
        print("enqueued")

    def subscribe(self, client: mqtt_client):

        client.subscribe(self.topic)
        client.on_message = self.on_message
        
    def run(self):
        client = self.connect_mqtt()
        self.subscribe(client)
        try:
            client.loop_forever()
        except KeyboardInterrupt as ex:
            print("Exiting")

            
def mqtt_entry(message_queue):
    subscriber = Subscriber(message_queue)
    subscriber.run()


class MyHandler(http.server.SimpleHTTPRequestHandler):

    def do_HEAD(self):
        print("do_HEAD")
        self.send_response(200)
        self.send_header("Content-type", "image/jpg")
        self.end_headers()

    def do_GET(self):
        print("do_GET")
        
        server = self.server
        try:
            msg_str = server.message_queue.get(timeout=0.5)
            print("dequeued")
        except Exception as ex:
            print("Empty queue", str(ex))
            msg_str = None
        
        if msg_str is not None:
            msg = json.loads(msg_str)
            frame_id = msg['frame_id']
            print("got msg", frame_id)
            image_base64 = msg['image_base64']
            shape = msg['shape']
            biggest = msg['biggest']
            print("biggest", biggest)
            nparr_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(nparr_bytes, dtype=np.uint8)
            nparr = nparr.reshape(shape)
            if biggest is not None:
                biggest = [int(v) for v in biggest]
                biggest = [shape[1]-biggest[0], biggest[1], shape[1]-biggest[2], biggest[3]]
                cv2.rectangle(nparr, tuple(biggest[0:2]), tuple(biggest[2:4]),
                              color=(0, 255, 0), thickness=1)
            success, jpg_bytes = cv2.imencode('.jpg', nparr)
            server.recent_jpg_bytes = jpg_bytes
        
        if server.recent_jpg_bytes is None:
            return

        self.send_response(200)
        self.send_header("Content-type", "image/jpg")
        self.send_header("Content-length", len(server.recent_jpg_bytes))
        self.end_headers() 
        
        self.wfile.write(server.recent_jpg_bytes)


class MyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    def __init__(self, server_adress, RequestHandlerClass, message_queue):
        self.allow_reuse_address = True
        self.message_queue = message_queue
        self.recent_jpg_bytes = None
        socketserver.TCPServer.__init__(self, server_adress, RequestHandlerClass, False)


def server_entry(message_queue):
    HOST, PORT = "localhost", 8080
    server = MyServer((HOST, PORT), MyHandler, message_queue)
    server.server_bind()
    server.server_activate()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    
    
def main():
    message_queue = Queue(maxsize=1)
    mqtt_thread = Thread(target=mqtt_entry, args=(message_queue,))
    mqtt_thread.start()

    server_thread = Thread(target=server_entry, args=(message_queue,))
    server_thread.start()

    try:
        server_thread.join()
        mqtt_thread.join()
    except KeyboardInterrupt:
        server_thread.terminate()
        mqtt_thread.terminate()

    
if __name__ == '__main__':
    main()
