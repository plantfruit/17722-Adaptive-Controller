import tkinter as tk
from threading import Thread
import socket
import pydirectinput as pydin

class SocketServerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Socket Server Control")

        self.keyLabels = ["Left", "Up", "Right", "Down", "Center"]

        self.server_thread = None
        self.running = False
        self.server = None
        self.client = None

        self.key_inputs = {}
        for i in range(0, 5):
            keyLabel = self.keyLabels[i]
            label = tk.Label(root, text=f"{keyLabel} Key:")
            label.grid(row=i, column=0)
            entry = tk.Entry(root)
            entry.insert(0, self.default_key(i))
            entry.grid(row=i, column=1)
            self.key_inputs[str(i)] = entry

        self.start_button = tk.Button(root, text="Start Server", command=self.start_server)
        self.start_button.grid(row=6, column=0)

        self.stop_button = tk.Button(root, text="Stop Server", command=self.stop_server, state=tk.DISABLED)
        self.stop_button.grid(row=6, column=1)

    def default_key(self, i):
        return ["left", "up", "right", "down", "space"][i - 1]

    def get_keybindings(self):
        mapping = {}
        for label, entry in self.key_inputs.items():
            mapping[label] = entry.get()
        mapping["6"] = "unpressed"
        mapping["7"] = "stop"
        return mapping

    def start_server(self):
        if not self.running:
            self.running = True
            self.server_thread = Thread(target=self.run_server, daemon=True)
            self.server_thread.start()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

    def stop_server(self):
        self.running = False
        if self.client:
            try:
                self.client.close()
            except:
                pass
        if self.server:
            try:
                self.server.close()
            except:
                pass
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def run_server(self):
        label_to_key = self.get_keybindings()
        N = 3
        consecutive_presses = {key: 0 for key in label_to_key.values()}
        pressed_key = "|"
        old_key = "|"

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("0.0.0.0", 5000))
        self.server.listen(1)

        print("Waiting for Android to connect...")
        try:
            self.client, addr = self.server.accept()
            print(f"Connected to {addr}")
        except:
            return

        while self.running:
            try:
                data = self.client.recv(1024).decode().strip()
                if not data:
                    continue
            except:
                break

            if data in label_to_key:
                key = label_to_key[data]
                print(f"Detected: {key}, Pressed: {pressed_key}")

                if key == 'stop':
                    self.running = False
                    break
                else:
                    if old_key == key:
                        consecutive_presses[key] += 1
                    else:
                        consecutive_presses[key] = 1
                        old_key = key
                    if consecutive_presses[key] >= N and pressed_key != key:
                        pydin.keyUp(pressed_key)
                        pydin.keyDown(key)
                        pressed_key = key

        pydin.keyUp(pressed_key)
        self.stop_server()

if __name__ == "__main__":
    root = tk.Tk()
    app = SocketServerApp(root)
    root.mainloop()
