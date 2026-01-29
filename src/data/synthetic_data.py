# Smart Industrial Defect Detection - Synthetic Data Generator
"""
Generate synthetic defect images for training.
Useful when real factory data is limited.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import random
import logging

logger = logging.getLogger(__name__)


class SyntheticDefectGenerator:
    """
    Generate synthetic defects on clean product images.
    Useful for data augmentation and initial model training.
    """
    
    def __init__(self, output_dir: str = 'data/synthetic'):
        """
        Initialize generator.
        
        Args:
            output_dir: Directory to save generated images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Defect generation parameters
        self.defect_params = {
            'scratch': {
                'length_range': (50, 200),
                'width_range': (1, 5),
                'color_range': ((100, 100, 100), (200, 200, 200))
            },
            'crack': {
                'num_segments': (3, 8),
                'segment_length': (20, 60),
                'width_range': (1, 3),
                'color_range': ((50, 50, 50), (150, 150, 150))
            },
            'dent': {
                'radius_range': (20, 80),
                'depth_range': (0.1, 0.3),  # Relative to max intensity
                'shadow_offset': (5, 15)
            },
            'missing_component': {
                'size_range': (30, 100),
                'shape': ['rectangle', 'circle', 'polygon']
            },
            'contamination': {
                'num_spots': (3, 15),
                'spot_size': (5, 30),
                'color_range': ((20, 20, 20), (80, 80, 80))
            }
        }
    
    def generate_scratch(self, image: np.ndarray,
                        num_scratches: int = 1) -> Tuple[np.ndarray, List[List]]:
        """
        Generate scratch defects on image.
        
        Args:
            image: Input image (H, W, C)
            num_scratches: Number of scratches to generate
        
        Returns:
            Image with scratches, list of bounding boxes
        """
        result = image.copy()
        bboxes = []
        h, w = image.shape[:2]
        
        for _ in range(num_scratches):
            # Random scratch parameters
            params = self.defect_params['scratch']
            length = random.randint(*params['length_range'])
            width = random.randint(*params['width_range'])
            
            # Random start point
            x1 = random.randint(0, w - length)
            y1 = random.randint(0, h - length)
            
            # Random angle
            angle = random.uniform(0, 2 * np.pi)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            # Clip to image bounds
            x2 = np.clip(x2, 0, w - 1)
            y2 = np.clip(y2, 0, h - 1)
            
            # Random color
            color = tuple(random.randint(params['color_range'][0][i], 
                                        params['color_range'][1][i]) for i in range(3))
            
            # Draw scratch with anti-aliasing
            cv2.line(result, (x1, y1), (x2, y2), color, width, lineType=cv2.LINE_AA)
            
            # Calculate bounding box (YOLO format: x_center, y_center, width, height)
            bbox_x = (x1 + x2) / 2 / w
            bbox_y = (y1 + y2) / 2 / h
            bbox_w = (abs(x2 - x1) + width * 2) / w
            bbox_h = (abs(y2 - y1) + width * 2) / h
            bboxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
        
        return result, bboxes
    
    def generate_crack(self, image: np.ndarray,
                      num_cracks: int = 1) -> Tuple[np.ndarray, List[List]]:
        """
        Generate crack defects on image.
        Cracks are jagged, branching lines.
        
        Args:
            image: Input image
            num_cracks: Number of cracks to generate
        
        Returns:
            Image with cracks, list of bounding boxes
        """
        result = image.copy()
        bboxes = []
        h, w = image.shape[:2]
        
        for _ in range(num_cracks):
            params = self.defect_params['crack']
            num_segments = random.randint(*params['num_segments'])
            width = random.randint(*params['width_range'])
            
            # Start point
            x, y = random.randint(50, w - 50), random.randint(50, h - 50)
            points = [(x, y)]
            min_x, max_x, min_y, max_y = x, x, y, y
            
            # Generate crack path
            angle = random.uniform(0, 2 * np.pi)
            for _ in range(num_segments):
                seg_length = random.randint(*params['segment_length'])
                angle += random.uniform(-0.5, 0.5)  # Small angle variation
                
                new_x = int(x + seg_length * np.cos(angle))
                new_y = int(y + seg_length * np.sin(angle))
                
                new_x = np.clip(new_x, 0, w - 1)
                new_y = np.clip(new_y, 0, h - 1)
                
                points.append((new_x, new_y))
                min_x, max_x = min(min_x, new_x), max(max_x, new_x)
                min_y, max_y = min(min_y, new_y), max(max_y, new_y)
                
                x, y = new_x, new_y
            
            # Draw crack
            color = tuple(random.randint(params['color_range'][0][i],
                                        params['color_range'][1][i]) for i in range(3))
            
            for i in range(len(points) - 1):
                cv2.line(result, points[i], points[i + 1], color, width, lineType=cv2.LINE_AA)
            
            # Calculate bounding box
            padding = 10
            bbox_x = (min_x + max_x) / 2 / w
            bbox_y = (min_y + max_y) / 2 / h
            bbox_w = (max_x - min_x + padding) / w
            bbox_h = (max_y - min_y + padding) / h
            bboxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
        
        return result, bboxes
    
    def generate_dent(self, image: np.ndarray,
                     num_dents: int = 1) -> Tuple[np.ndarray, List[List]]:
        """
        Generate dent defects on image.
        Dents are circular shadows with highlights.
        
        Args:
            image: Input image
            num_dents: Number of dents to generate
        
        Returns:
            Image with dents, list of bounding boxes
        """
        result = image.copy().astype(np.float32)
        bboxes = []
        h, w = image.shape[:2]
        
        for _ in range(num_dents):
            params = self.defect_params['dent']
            radius = random.randint(*params['radius_range'])
            depth = random.uniform(*params['depth_range'])
            shadow_offset = random.randint(*params['shadow_offset'])
            
            # Random center
            cx = random.randint(radius + shadow_offset, w - radius - shadow_offset)
            cy = random.randint(radius + shadow_offset, h - radius - shadow_offset)
            
            # Create dent effect (shadow on one side, highlight on other)
            y_grid, x_grid = np.ogrid[:h, :w]
            dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            
            # Gradient within dent area
            mask = dist < radius
            gradient = (radius - dist) / radius
            gradient = np.clip(gradient, 0, 1)
            
            # Apply shadow
            shadow_factor = 1 - depth * gradient
            result[mask] = result[mask] * shadow_factor[mask, np.newaxis]
            
            # Add highlight on edge
            edge_mask = (dist > radius * 0.7) & (dist < radius)
            highlight = 1 + 0.2 * gradient
            result[edge_mask] = np.clip(result[edge_mask] * highlight[edge_mask, np.newaxis], 0, 255)
            
            # Bounding box
            bbox_x = cx / w
            bbox_y = cy / h
            bbox_w = (radius * 2 + shadow_offset) / w
            bbox_h = (radius * 2 + shadow_offset) / h
            bboxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
        
        return result.astype(np.uint8), bboxes
    
    def generate_missing_component(self, image: np.ndarray,
                                   num_missing: int = 1) -> Tuple[np.ndarray, List[List]]:
        """
        Generate missing component defects.
        Represents holes or missing parts in product.
        
        Args:
            image: Input image
            num_missing: Number of missing regions
        
        Returns:
            Image with missing components, list of bounding boxes
        """
        result = image.copy()
        bboxes = []
        h, w = image.shape[:2]
        
        for _ in range(num_missing):
            params = self.defect_params['missing_component']
            size = random.randint(*params['size_range'])
            shape = random.choice(params['shape'])
            
            # Random position
            cx = random.randint(size, w - size)
            cy = random.randint(size, h - size)
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            if shape == 'rectangle':
                x1, y1 = cx - size // 2, cy - size // 2
                x2, y2 = cx + size // 2, cy + size // 2
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            elif shape == 'circle':
                cv2.circle(mask, (cx, cy), size // 2, 255, -1)
            else:  # polygon
                num_vertices = random.randint(5, 8)
                angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
                angles += random.uniform(0, np.pi / 4)
                radii = [size // 2 + random.randint(-size // 4, size // 4) for _ in angles]
                points = [(int(cx + r * np.cos(a)), int(cy + r * np.sin(a))) 
                         for r, a in zip(radii, angles)]
                cv2.fillPoly(mask, [np.array(points)], 255)
            
            # Apply mask (darken area)
            result[mask > 0] = (result[mask > 0] * 0.3).astype(np.uint8)
            
            # Bounding box
            bbox_x = cx / w
            bbox_y = cy / h
            bbox_w = size / w
            bbox_h = size / h
            bboxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
        
        return result, bboxes
    
    def generate_contamination(self, image: np.ndarray,
                              num_regions: int = 1) -> Tuple[np.ndarray, List[List]]:
        """
        Generate contamination/stain defects.
        Random spots or blobs on surface.
        
        Args:
            image: Input image
            num_regions: Number of contamination regions
        
        Returns:
            Image with contamination, list of bounding boxes
        """
        result = image.copy()
        bboxes = []
        h, w = image.shape[:2]
        
        for _ in range(num_regions):
            params = self.defect_params['contamination']
            num_spots = random.randint(*params['num_spots'])
            
            # Cluster center
            cluster_x = random.randint(50, w - 50)
            cluster_y = random.randint(50, h - 50)
            cluster_radius = random.randint(30, 80)
            
            min_x, max_x = cluster_x, cluster_x
            min_y, max_y = cluster_y, cluster_y
            
            for _ in range(num_spots):
                # Random spot position within cluster
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(0, cluster_radius)
                spot_x = int(cluster_x + dist * np.cos(angle))
                spot_y = int(cluster_y + dist * np.sin(angle))
                
                spot_x = np.clip(spot_x, 0, w - 1)
                spot_y = np.clip(spot_y, 0, h - 1)
                
                spot_size = random.randint(*params['spot_size'])
                color = tuple(random.randint(params['color_range'][0][i],
                                            params['color_range'][1][i]) for i in range(3))
                
                # Draw spot with blur for natural look
                cv2.circle(result, (spot_x, spot_y), spot_size, color, -1)
                
                min_x = min(min_x, spot_x - spot_size)
                max_x = max(max_x, spot_x + spot_size)
                min_y = min(min_y, spot_y - spot_size)
                max_y = max(max_y, spot_y + spot_size)
            
            # Apply slight blur to make it look more natural
            roi = result[max(0, min_y):min(h, max_y), max(0, min_x):min(w, max_x)]
            if roi.size > 0:
                result[max(0, min_y):min(h, max_y), max(0, min_x):min(w, max_x)] = \
                    cv2.GaussianBlur(roi, (5, 5), 0)
            
            # Bounding box
            bbox_x = (min_x + max_x) / 2 / w
            bbox_y = (min_y + max_y) / 2 / h
            bbox_w = (max_x - min_x) / w
            bbox_h = (max_y - min_y) / h
            bboxes.append([bbox_x, bbox_y, min(bbox_w, 1), min(bbox_h, 1)])
        
        return result, bboxes
    
    def generate_dataset(self, 
                        clean_images_dir: str,
                        num_samples_per_class: int = 100,
                        defect_types: List[str] = None) -> Dict:
        """
        Generate a complete synthetic dataset.
        
        Args:
            clean_images_dir: Directory with clean product images
            num_samples_per_class: Number of samples per defect class
            defect_types: List of defect types to generate
        
        Returns:
            Dictionary with generation statistics
        """
        if defect_types is None:
            defect_types = ['scratch', 'crack', 'dent', 'missing_component', 'contamination']
        
        clean_dir = Path(clean_images_dir)
        clean_images = list(clean_dir.glob('*.jpg')) + list(clean_dir.glob('*.png'))
        
        if not clean_images:
            logger.warning(f"No clean images found in {clean_images_dir}")
            return {'status': 'error', 'message': 'No clean images found'}
        
        stats = {defect: 0 for defect in defect_types}
        
        defect_generators = {
            'scratch': self.generate_scratch,
            'crack': self.generate_crack,
            'dent': self.generate_dent,
            'missing_component': self.generate_missing_component,
            'contamination': self.generate_contamination
        }
        
        for defect_type in defect_types:
            output_images_dir = self.output_dir / 'images' / defect_type
            output_labels_dir = self.output_dir / 'labels' / defect_type
            output_images_dir.mkdir(parents=True, exist_ok=True)
            output_labels_dir.mkdir(parents=True, exist_ok=True)
            
            class_id = defect_types.index(defect_type)
            generator = defect_generators[defect_type]
            
            for i in range(num_samples_per_class):
                # Random clean image
                clean_img_path = random.choice(clean_images)
                clean_img = cv2.imread(str(clean_img_path))
                
                if clean_img is None:
                    continue
                
                # Generate defect
                num_defects = random.randint(1, 3)
                defect_img, bboxes = generator(clean_img, num_defects)
                
                # Save image
                output_img_path = output_images_dir / f'{defect_type}_{i:04d}.jpg'
                cv2.imwrite(str(output_img_path), defect_img)
                
                # Save label (YOLO format)
                output_label_path = output_labels_dir / f'{defect_type}_{i:04d}.txt'
                with open(output_label_path, 'w') as f:
                    for bbox in bboxes:
                        f.write(f'{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n')
                
                stats[defect_type] += 1
        
        logger.info(f"Generated synthetic dataset: {stats}")
        return {'status': 'success', 'samples_generated': stats}


def generate_clean_product_images(output_dir: str,
                                  num_images: int = 100,
                                  image_size: Tuple[int, int] = (640, 640)) -> int:
    """
    Generate clean product images (simulated metal/plastic surfaces).
    Useful when no real product images are available.
    
    Args:
        output_dir: Output directory
        num_images: Number of images to generate
        image_size: Image dimensions
    
    Returns:
        Number of images generated
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        # Generate base surface
        surface_type = random.choice(['metal', 'plastic', 'wood', 'glass'])
        
        if surface_type == 'metal':
            # Brushed metal texture
            base_color = random.randint(150, 220)
            image = np.ones((*image_size, 3), dtype=np.uint8) * base_color
            
            # Add brushed texture
            noise = np.random.normal(0, 10, image_size).astype(np.float32)
            noise = cv2.GaussianBlur(noise, (1, 21), 0)  # Directional blur
            image = np.clip(image + noise[:, :, np.newaxis], 0, 255).astype(np.uint8)
            
        elif surface_type == 'plastic':
            # Smooth plastic surface
            base_color = random.randint(180, 255)
            image = np.ones((*image_size, 3), dtype=np.uint8) * base_color
            
            # Add subtle variation
            noise = np.random.normal(0, 5, image_size).astype(np.float32)
            image = np.clip(image + noise[:, :, np.newaxis], 0, 255).astype(np.uint8)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            
        elif surface_type == 'wood':
            # Wood grain texture
            base_color = (139, 90, 43)  # Brown
            image = np.zeros((*image_size, 3), dtype=np.uint8)
            image[:, :] = base_color
            
            # Add grain pattern
            for _ in range(50):
                y = random.randint(0, image_size[0] - 1)
                color_var = random.randint(-30, 30)
                line_color = tuple(np.clip(np.array(base_color) + color_var, 0, 255).astype(int))
                cv2.line(image, (0, y), (image_size[1], y + random.randint(-10, 10)), 
                        line_color, random.randint(1, 3))
            image = cv2.GaussianBlur(image, (3, 3), 0)
            
        else:  # glass
            # Glass/transparent surface
            base_color = random.randint(200, 255)
            image = np.ones((*image_size, 3), dtype=np.uint8) * base_color
            
            # Add reflections
            for _ in range(5):
                x, y = random.randint(0, image_size[1]), random.randint(0, image_size[0])
                radius = random.randint(50, 150)
                cv2.circle(image, (x, y), radius, (255, 255, 255), -1)
            image = cv2.GaussianBlur(image, (51, 51), 0)
        
        # Save image
        output_file = output_path / f'clean_{surface_type}_{i:04d}.jpg'
        cv2.imwrite(str(output_file), image)
    
    logger.info(f"Generated {num_images} clean product images in {output_dir}")
    return num_images
