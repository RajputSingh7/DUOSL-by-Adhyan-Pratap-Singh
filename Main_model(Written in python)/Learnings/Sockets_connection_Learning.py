import socket
import cv2
import struct

HOST = "127.0.0.1"
PORT1 = 5005 # Sign port
PORT2 = 5001 # Camera port

# Create socket
server_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket1.bind((HOST, PORT1))
server_socket1.listen(1)
server_socket2.bind((HOST, PORT2))
server_socket2.listen(1)


print("Waiting for Unity connection...")
conn2, addr2 = server_socket2.accept()
print("Connected to:", addr2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    frame = cv2.resize(frame, (1280, 720))

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    data = buffer.tobytes()

    # Send length + data
    message = struct.pack("Q", len(data)) + data
    conn2.sendall(message)

cap.release()
conn2.close()
