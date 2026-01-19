import socket
from threading import Thread

HOST = '0.0.0.0'
PORT = 21001

clients = {}
messages = []
next_client_id = 1


def client_processor(conn, client_id):
    global clients, messages

    conn.send(f"Your client ID is {client_id}\n".encode())

    while True:
        data = conn.recv(4096)
        if not data:
            print(f"Client {client_id} disconnected")
            del clients[client_id]
            break

        text = data.decode().strip()
        print(f"Received from {client_id}: {text}")

        recipients = [cid for cid in clients if cid != client_id]

        messages.append({
            "from": client_id,
            "need_to_send": recipients,
            "text": text
        })

        outgoing = []

        for msg in messages:
            if client_id in msg["need_to_send"]:
                outgoing.append(f"Client {msg['from']}: {msg['text']}")
                msg["need_to_send"].remove(client_id)

        if outgoing:
            conn.send(("\n".join(outgoing) + "\n").encode())
        else:
            conn.send("No new messages for you for now!\n".encode())


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
print("Server listening")

while True:
    conn, addr = s.accept()
    client_id = next_client_id
    next_client_id += 1

    clients[client_id] = conn
    print(f"Client {client_id} connected from {addr}")

    Thread(target=client_processor, args=(conn, client_id), daemon=True).start()
