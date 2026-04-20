import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def visualize_side_by_side(img1: np.ndarray, img2: np.ndarray, title1: str = "Image 1", title2: str = "Image 2") -> None:
    """Visualizes two images side by side using matplotlib."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, img, title in zip(axes, [img1, img2], [title1, title2]):
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_triple(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray, 
                     title1: str = "Image 1", title2: str = "Image 2", title3: str = "Image 3") -> None:
    """
    Visualizes three images side by side in one window.
    
    Args:
        img1, img2, img3 (np.ndarray): Image arrays to visualize.
        title1, title2, title3 (str): Titles for each subplot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(axes, [img1, img2, img3], [title1, title2, title3]):
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_overlay(img1: np.ndarray, img2: np.ndarray, title1: str = "Render", title2: str = "Real") -> None:
    """Visualizes an interactive overlay of two images with an alpha slider."""
    if img1.shape != img2.shape:
        from PIL import Image
        img1 = np.array(Image.fromarray(img1).resize((img2.shape[1], img2.shape[0]), Image.Resampling.BILINEAR))

    i1_float = img1.astype(np.float32) / 255.0
    i2_float = img2.astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.2)
    
    blend_ax = axes[0]
    initial_alpha = 0.5
    blended = (initial_alpha * i1_float + (1.0 - initial_alpha) * i2_float)
    img_display = blend_ax.imshow(np.clip(blended, 0, 1))
    blend_ax.set_title(f"Interactive Overlay (Alpha Blend)")
    blend_ax.axis('off')
    
    diff_ax = axes[1]
    diff = np.abs(i1_float - i2_float)
    diff_gray = np.mean(diff, axis=2) if diff.ndim == 3 else diff
    diff_ax.imshow(diff_gray, cmap='magma')
    diff_ax.set_title("Absolute Difference")
    diff_ax.axis('off')
    
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, f'{title2} <---> {title1}', 0.0, 1.0, valinit=initial_alpha)
    
    def update(val):
        alpha = slider.val
        new_blend = (alpha * i1_float + (1.0 - alpha) * i2_float)
        img_display.set_data(np.clip(new_blend, 0, 1))
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    plt.show()

def visualize_matches(img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, title: str = "Matches") -> None:
    """
    Visualizes feature matches between two images side by side.
    
    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        pts1 (np.ndarray): N x 2 array of points in img1.
        pts2 (np.ndarray): N x 2 array of points in img2.
        title (str): Title of the plot.
    """
    import cv2
    if img1.shape != img2.shape:
        from PIL import Image
        img1 = np.array(Image.fromarray(img1).resize((img2.shape[1], img2.shape[0]), Image.Resampling.BILINEAR))

    # Convert to uint8 if they are float
    if img1.dtype != np.uint8:
        img1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)

    # Ensure RGB
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    # Create Keypoints
    kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in pts1]
    kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=10) for pt in pts2]

    # Create DMatches
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(pts1))]

    # Draw matches
    out_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                              matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                              flags=cv2.DrawMatchesFlags_DEFAULT)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(out_img)
    ax.set_title(f"{title} - {len(pts1)} matches")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
