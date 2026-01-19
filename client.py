import socket

HOST = '127.0.0.1'
PORT = 21001

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    client_id = s.recv(1024).decode().strip()
    print(client_id)

    while True:
        message = input("Enter message: ")
        s.send(message.encode())

        data = s.recv(4096)
        if not data:
            print("Server disconnected")
            break

        print(data.decode())
