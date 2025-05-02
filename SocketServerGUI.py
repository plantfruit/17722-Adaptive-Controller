import tkinter as tk
import tkinter.font as tkFont
from threading import Thread
import socket
import pydirectinput as pydin
import threading

class SocketServerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Socket Server Control")
        self.root.geometry("800x500")  # Initial size
        self.stop_event = threading.Event()
        self.server_thread = None

        # Define the d-pad names (used to label the keybinding inputs)
        self.keyLabels = ["Left", "Up", "Right", "Down", "Center"]
        self.defaultKeys = ["left", "up", "right", "down", "space"]

        self.running = False
        self.server = None
        self.client = None

        # Make entire grid expandable
        for i in range(8):
            root.grid_rowconfigure(i, weight=1)
        for j in range(2):  
            root.grid_columnconfigure(j, weight=1, uniform = "col")

        # Set font parameters
        self.default_font = tkFont.Font(family="Helvetica", size=14)

        # Labels
        # Keybinding 
        self.key_inputs = {}
        for i in range(1, 6):
            keyLabel = self.keyLabels[i-1]
            
            label = tk.Label(root, text=f"{keyLabel} Key:", font = self.default_font)
            label.grid(row = i, column = 0, sticky = "nsew", padx = 10, pady = 5)
            
            entry = tk.Entry(root, font = self.default_font)
            entry.insert(0, self.defaultKeys[i-1])
            entry.grid(row = i, column = 1, sticky = "nsew", padx = 10, pady = 5)            
            self.key_inputs[str(i)] = entry
        # Title
        title_label = tk.Label(root, text="Socket Server Controller", font=("Helvetica", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(15, 10), sticky="n")
        
        # Server buttons
        #self.start_button = tk.Button(root, text="Start Server", command=self.start_server, font = self.default_font)
        #self.start_button.grid(row=6, column=0, sticky = "nsew", padx = 10, pady = 5)

        #self.stop_button = tk.Button(root, text="Stop Server", command=self.stop_server, state=tk.DISABLED, font = self.default_font)
        #self.stop_button.grid(row=6, column=1, sticky = "nsew", padx = 10, pady = 5)


        self.start_button = tk.Button(root, text="Start Server", font=self.default_font, command=self.start_server)
        self.stop_button = tk.Button(root, text="Stop Server", font=self.default_font, command=self.stop_server)
        self.quit_button = tk.Button(root, text="Quit App", font=self.default_font, command=self.quit_app)

        self.start_button.grid(row=6, column=0, sticky="nsew", padx=5)
        self.stop_button.grid(row=6, column=1, sticky="nsew", padx=5)
        self.quit_button.grid(row=7, column=0, columnspan = 2, sticky="nsew", padx=5)

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

    def quit_app(self):
        print("Quitting application...")
        self.stop_server()
        self.stop_event.set()
        self.root.destroy()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = SocketServerApp(root)
    root.mainloop()
