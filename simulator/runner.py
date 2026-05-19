import torch
import argparse
import json
import random
import socket
import sys
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

ROOT = Path(__file__).resolve().parents[1]
for module_path in (ROOT, ROOT / "src", ROOT / "tasks" / "task2-detector" / "src"):
    path_str = str(module_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import cv2
import numpy as np

from simulator_client.pipeline import FallbackPipeline
from simulator_client.protocol import (
    ConfigMessage,
    EndMessage,
    FrameMessage,
    parse_message,
    start_payload,
    write_payload,
)


HOST = "127.0.0.1"
PORT = 50000
LATENCY = 0.50
CAMERA_MATRIX = [
    960.0, 0.0, 640.0,
    0.0, 960.0, 360.0,
    0.0, 0.0, 1.0,
]


def decode_image_to_rgb_pixels(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("OpenCV failed to decode simulator frame")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the simulator client.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Use a fixed simulator seed. Defaults to a random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    with socket.create_connection((HOST, PORT)) as sock:
        sock_file = sock.makefile("rw", encoding="utf-8", newline="\n")

        camera_matrix = (
            (CAMERA_MATRIX[0], CAMERA_MATRIX[1], CAMERA_MATRIX[2]),
            (CAMERA_MATRIX[3], CAMERA_MATRIX[4], CAMERA_MATRIX[5]),
            (CAMERA_MATRIX[6], CAMERA_MATRIX[7], CAMERA_MATRIX[8]),
        )

        write_payload(
            sock_file,
            start_payload(seed, LATENCY, camera_matrix),
        )

        pipeline = None

        while True:
            line = sock_file.readline()
            if not line:
                print("Connection closed by server.")
                break

            message = parse_message(json.loads(line))

            if isinstance(message, ConfigMessage):
                pipeline = FallbackPipeline(
                    latency=LATENCY,
                    target_depth=10.0,
                    board_width_meters=message.board_width,
                    board_height_meters=message.board_height,
                )
                print(
                    f"Game configured: latency={LATENCY} s, "
                    f"board={message.board_width}x{message.board_height} m"
                )
            elif isinstance(message, FrameMessage):
                if pipeline is None:
                    print("Received frame before config, skipping.")
                    continue
                image = decode_image_to_rgb_pixels(message.image_bytes)
                result = pipeline.process_rgb_image(image, camera_matrix, message.timestamp)
                write_payload(sock_file, result.aim.to_payload())
            elif isinstance(message, EndMessage):
                print(f"Game ended. Final score: {message.score}, accuracy: {message.accuracy}, precision: {message.average_to_center_distance}")
                break


if __name__ == "__main__":
    main()
