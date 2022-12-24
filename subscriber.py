import cv2
import zmq
import numpy as np
BLOCK_SIZE = 32
port = "5555"

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.connect ("tcp://localhost:%s" % port)
# footage_socket.connect ("tcp://raspberrypi.local:%s" % port)
footage_socket.setsockopt_string(zmq.SUBSCRIBE, '')

def subscriber():
    while True:
        try:
            frame = footage_socket.recv()
            npimg = np.frombuffer(frame, dtype=np.uint8)
            source = cv2.imdecode(npimg, 1)
            # source = cv2.cvtColor(source , cv2.COLOR_BGR2RGB)
            cv2.imshow("Stream", source)
            cv2.waitKey(1)

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break

subscriber()