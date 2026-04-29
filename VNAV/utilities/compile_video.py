import cv2
import os

def images_to_video(image_folder, output_path, fps=30):
    # Get all image files and sort them
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    images = sorted([img for img in os.listdir(image_folder) if img.lower().endswith(valid_ext)])

    if not images:
        print("No images found in the directory.")
        return

    # Read first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)

    if frame is None:
        print("Error reading the first image.")
        return

    height, width, _ = frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # .mp4 format
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Skipping unreadable image: {image}")
            continue

        # Resize if needed to match first frame
        frame = cv2.resize(frame, (width, height))

        video.write(frame)

    video.release()
    print(f"Video saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    folder = "/home/haytam-elourrat/VISNAV/RUNS/20260428_200058/frames/"
    output_video = "/home/haytam-elourrat/VISNAV/RUNS/20260428_200058/output.mp4"
    fps = 24

    images_to_video(folder, output_video, fps)
