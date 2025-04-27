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

# Number of consecutive presses required
N = 3  # You can adjust this value as needed

# Dictionary to track consecutive presses for each key
consecutive_presses = {key: 0 for key in label_to_key.values()}

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#server.bind(("127.0.0.1", 5000))
server.bind(("0.0.0.0", 5000))
server.listen(1)

print("Waiting for Android to connect...")
client, addr = server.accept()
print(f"Connected to {addr}")

loopRunning = True
pressed_key = "|"
old_key = "|"
while loopRunning:
    data = client.recv(1024).decode().strip()
    #print(data)

    if data in label_to_key:        
        key = label_to_key[data]
        print(f"Pressed: {key}")

        if (key == 'stop'):
            loopRunning = False
        else:
            if old_key == key:
                consecutive_presses[key] += 1
            else:
                consecutive_presses[key] = 1
                old_key = key

            if consecutive_presses[key] >= N:
                pydin.keyUp(pressed_key)
                pydin.keyDown(key)
                pressed_key = key
