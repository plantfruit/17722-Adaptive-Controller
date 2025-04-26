import socket
from pynput.keyboard import Controller
import keyboard
import pydirectinput as pydin

#keyboard = Controller()

label_to_key = {
    "1": 'a',    # left
    "2": 'w',    # up
    "3": 'd',    # right
    "4": 's',    # down
    "5": 'space',
    "6": 'pause',
    "7": 'stop',
}

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#server.bind(("127.0.0.1", 5000))
server.bind(("0.0.0.0", 5000))
server.listen(1)

print("Waiting for Android to connect...")
client, addr = server.accept()
print(f"Connected to {addr}")

loopRunning = True
oldKey = "|"
while loopRunning:
    data = client.recv(1024).decode().strip()
    #print(data)

    if data in label_to_key:        
        key = label_to_key[data]
        #print(f"Pressed: {key}")

        if (key == 'stop'):
            loopRunning = False
        elif(key == 'pause'):
            pydin.keyUp(oldKey)
            oldKey = 'pause'
        elif (not(oldKey == key)):
            pydin.keyUp(oldKey)
            pydin.keyDown(key)
            oldKey = key
        #keyboard.press(key)
        #keyboard.release(key)
