import asyncio
import websockets
import json
import base64


async def save_image(websocket):
    print(
        f"Client Connected: IP: {websocket.remote_address[0]}, Port: {websocket.remote_address[1]}")

    message_count = 0

    async for message in websocket:
        message_count += 1

        try:
            data = json.loads(message)
            image_data = data['imageData']
            diopter_value = data['focusDistance']

            image_bytes = base64.b64decode(image_data)

            # dir_name = f"stack_{message_count}"
            # os.makedirs(dir_name, exist_ok=True)
            # Convert diopter values to meters: distance_in_meters = 1 / diopter_value
            focus_distance = 1 / diopter_value if diopter_value != 0 else 0
            # focus_distance = diopter_value
            

            filename = f"./imgs/image_{message_count}_{focus_distance}.png"
            with open(filename, 'wb') as img_file:
                img_file.write(image_bytes)

            print(f"Saved {filename} with focus distance: {focus_distance}")

        except json.JSONDecodeError:
            print("Received non-JSON message, ignoring.")
        except KeyError:
            print("JSON message is missing required data fields.")
        except Exception as e:
            print(f"An error occurred: {e}")


async def main():
    max_size = 20 * 1024 * 1024  # Increase the max_size to 20 MB
    url = "0.0.0.0"
    port = 8765
    async with websockets.serve(save_image, url, port, max_size=max_size):
        print(f"Server started. Listening on http://{url}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

   