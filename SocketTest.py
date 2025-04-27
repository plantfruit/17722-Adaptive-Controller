import socket
#from pynput.keyboard import Controller
#import keyboard
import pydirectinput as pydin

#keyboard = Controller()

label_to_key = {
    "1": 'left', #'a',    # left
    "2": 'up', #'w',    # up
    "3": 'right', #'d',    # right
    "4": 'down', #'s',    # down
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
         

        if (key == 'stop'):
            loopRunning = False
        elif(key == 'pause'):
            pydin.keyUp(oldKey)
            oldKey = 'pause'
        elif (not(oldKey == key)):
            pydin.keyUp(oldKey)
            pydin.keyDown(key)
            oldKey = key

        print(f"Pressed: {key}")
        #keyboard.press(key)
        #keyboard.release(key)
