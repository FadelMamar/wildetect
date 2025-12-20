import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd 
from sklearn.cluster import MiniBatchKMeans  
from sklearn.metrics import silhouette_score
from collections import defaultdict
import random
import torch 
from tqdm import tqdm

from ...utils.logging import get_logger
from ...models.feature_extractor import FeatureExtractor
from ...utils.io import save_yaml,load_yaml,read_image
from ..curriculum.dataset import PatchDataset 


logger = get_logger(__name__)


# --- Sampling Strategy Base and Implementation ---
class SamplingStrategy:
    def sample(self, cluster_df: pd.DataFrame, n_samples: int) -> List[int]:
        raise NotImplementedError

class BaseFilter(ABC):
    """
    Abstract base class for all filters.
    """

    @abstractmethod
    def __call__(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the filter to the input COCO data and return filtered data.
        """
        pass

    def get_filter_info(self) -> Dict[str, Any]:
        """
        Return information about the filter (for logging/debugging).
        """
        return {"filter_type": self.__class__.__name__}
    
class UniformDistanceRandomSamplingStrategy(SamplingStrategy):
    def sample(self, cluster_df: pd.DataFrame, n_samples: int) -> List[int]:
        # Sort by distance to centroid
        cluster_df = cluster_df.sort_values("dist_to_centroid")
        if n_samples >= len(cluster_df):
            return cluster_df.index.tolist()
        # Split into bins and randomly select one from each bin
        bins = np.array_split(cluster_df, n_samples)
        chosen = []
        for b in bins:
            if isinstance(b, pd.DataFrame) and not b.empty:
                chosen.append(
                    b.sample(1, random_state=random.randint(0, 1_000_000)).index[0]
                )
        return chosen


class ClusteringFilter(BaseFilter):
    """
    Clustering-based filter. Ensures embeddings are present in each image as image['_embedding'].
    If not present, computes embeddings using FeatureExtractor and adds them in batches to avoid memory issues.
    Performs clustering on these embeddings and samples uniformly by distance to centroid, proportional to cluster size and x% reduction target.
    Sampling strategy is pluggable.
    """

    CLUSTER_TRIALS = [3, 5, 7, 9,]

    last_silhouette_scores: Optional[Dict[int, float]] = None
    last_samples_per_cluster: Optional[List[int]] = None

    def __init__(
        self,
        feature_extractor: FeatureExtractor = FeatureExtractor(),
        batch_size: int = 64,
        sampling_strategy: SamplingStrategy = UniformDistanceRandomSamplingStrategy(),
        reduction_factor: float = 0.3
    ):
        self.feature_extractor = FeatureExtractor()
        self.batch_size = batch_size
        self.x_percent = reduction_factor  # Default to 30% if not set
        self.sampling_strategy = (
            UniformDistanceRandomSamplingStrategy()
        )
        self.last_silhouette_scores = None
        self.last_samples_per_cluster = None

    def __call__(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not images:
            logger.warning("No images found in images")
            return images

        logger.info(f"Adding embeddings to {len(images)} images")
        images = self.add_embeddings(images)

        embeddings = np.stack([img["_embedding"] for img in images])
        image_ids = [img["id"] for img in images]
        file_names = [img["file_name"] for img in images]
        # Find best clustering
        best_k, best_labels, silhouette_scores = self._find_best_kmeans(embeddings)
        self.last_silhouette_scores = silhouette_scores
        logger.info(f"Best number of clusters: {best_k}")
        logger.info(f"Silhouette scores for tried clusters: {silhouette_scores}")
        # Create DataFrame for sampling
        df = pd.DataFrame(
            {
                "image_id": image_ids,
                "file_name": file_names,
                "cluster": best_labels,
                "embedding": embeddings.tolist(),
            }
        )  # type: ignore
        # Compute centroids
        kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42).fit(embeddings)
        centroids = kmeans.cluster_centers_
        df["dist_to_centroid"] = [
            np.linalg.norm(emb - centroids[cl])
            for emb, cl in zip(embeddings, best_labels)
        ]  # type: ignore
        # Log distance distribution
        dist_series = pd.Series(df["dist_to_centroid"])
        logger.info(
            f"Distance to centroid stats: min={dist_series.min():.4f}, max={dist_series.max():.4f}, mean={dist_series.mean():.4f}, std={dist_series.std():.4f}"
        )
        # Determine number of samples to keep
        n_total = len(df)
        n_keep = int(np.ceil(self.x_percent * n_total))
        logger.info(
            f"Target: keeping {n_keep} out of {n_total} images ({self.x_percent*100:.1f}%)"
        )
        # Proportional allocation
        samples_per_cluster = self._allocate_samples_per_cluster(df, n_keep)
        self.last_samples_per_cluster = samples_per_cluster
        logger.info(f"Samples per cluster: {samples_per_cluster}")
        selected_indices = []
        for cl in range(best_k):
            cluster_df = df[df["cluster"] == cl].copy()
            n_samples = samples_per_cluster[cl]
            if n_samples == 0:
                continue
            chosen = self.sampling_strategy.sample(cluster_df, n_samples)
            selected_indices.extend(chosen)
        # Filter images and annotations
        selected_image_ids = set(df.loc[selected_indices, "image_id"])  # type: ignore
        filtered_images = [
            img for img in images if img["id"] in selected_image_ids
        ]
        
        # Clean up embeddings before export
        for img in filtered_images:
            if "_embedding" in img:
                del img["_embedding"]
        return filtered_images

    def _find_best_kmeans(
        self, embeddings: np.ndarray
    ) -> Tuple[int, np.ndarray, Dict[int, float]]:
        best_score = -1
        best_k = self.CLUSTER_TRIALS[0]
        best_labels = None
        silhouette_scores = {}
        for k in self.CLUSTER_TRIALS:
            if k >= len(embeddings):
                continue
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42).fit(embeddings)
            labels = kmeans.labels_
            if len(set(labels)) == 1:
                continue  # skip degenerate case
            score = silhouette_score(embeddings, labels)
            silhouette_scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        return best_k, best_labels, silhouette_scores  # type: ignore

    def _allocate_samples_per_cluster(
        self, df: pd.DataFrame, n_keep: int,
    ) -> List[int]:
        # Proportional allocation, at least 1 per cluster if possible
        cluster_sizes = df["cluster"].value_counts().sort_index().values
        total = cluster_sizes.sum()
        base = np.floor(cluster_sizes / total * n_keep).astype(int) + 1 
        return base.tolist()  

    def add_embeddings(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(images) == 0:
            logger.warning("No images found in images")
            return images
        
        image_paths = [
            Path(img["file_name"]).resolve() for img in images
        ]
        embeddings = self._compute_embeddings_in_batches(image_paths)
        for img, emb in zip(images, embeddings):  
            img["_embedding"] = emb
        return images

    def _compute_embeddings_in_batches(
        self, image_paths: List[Union[str, Path]]
    ) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(image_paths), self.batch_size),unit="batch",desc="Computing embeddings"):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_images = [read_image(path) for path in batch_paths]
            batch_embeddings = self.feature_extractor(batch_images)  # type: ignore
            all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)

    def get_filter_info(self) -> Dict[str, Any]:
        return {"filter_type": self.__class__.__name__}


class CropClusteringAdapter:
    """
    Adapter to make ClusteringFilter work with PatchDataset annotations.
    
    This adapter converts between the PatchDataset annotation format and the
    ClusteringFilter's expected image format, allowing the ClusteringFilter
    to work with crop annotations without modification.
    """
    
    def __init__(self, clustering_filter: ClusteringFilter):
        """
        Initialize the adapter with a ClusteringFilter instance.
        
        Args:
            clustering_filter: The ClusteringFilter to adapt
        """
        self.clustering_filter = clustering_filter
    
    def __call__(self, crop_annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply clustering filter to crop annotations using adapter pattern.
        
        Args:
            crop_annotations: List of crop annotation dictionaries from PatchDataset
            
        Returns:
            Filtered list of crop annotation dictionaries
        """
        if not crop_annotations:
            logger.warning("No crop annotations found")
            return []
        
        # Convert crop annotations to ClusteringFilter format
        images_for_filter = self._adapt_crop_annotations_to_images(crop_annotations)
        
        # Apply the clustering filter
        filtered_images = self.clustering_filter(images_for_filter)
        
        # Convert back to crop annotation format
        filtered_annotations = self._adapt_images_to_crop_annotations(
            filtered_images, crop_annotations
        )
        
        return filtered_annotations
    
    def _adapt_crop_annotations_to_images(self, crop_annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert crop annotations to the format expected by ClusteringFilter.
        
        Args:
            crop_annotations: List of crop annotation dictionaries
            
        Returns:
            List of image dictionaries for ClusteringFilter
        """
        images = []
        
        for ann in crop_annotations:
            # Create image dict with required fields for ClusteringFilter
            image = {
                "id": ann["roi_id"],  # Use roi_id as image id
                "file_name": ann["file_name"],  # Use the generated filename
                "class_name": ann["class_name"],
                "class_id": ann["class_id"],
                "dataset_idx": ann["dataset_idx"],
                "crop_bbox": ann["crop_bbox"],
                "crop_type": ann["crop_type"],
                "_crop_info": ann  # Store the original crop info for later extraction
            }
            
            # Add original annotation info if available
            if "original_bbox" in ann:
                image["original_bbox"] = ann["original_bbox"]
            if "original_annotation_id" in ann:
                image["original_annotation_id"] = ann["original_annotation_id"]
            
            images.append(image)
        
        return images
    
    def _adapt_images_to_crop_annotations(
        self, filtered_images: List[Dict[str, Any]], original_annotations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert filtered images back to crop annotation format.
        
        Args:
            filtered_images: List of filtered image dictionaries from ClusteringFilter
            original_annotations: Original crop annotations for reference
            
        Returns:
            List of filtered crop annotation dictionaries
        """
        # Create mapping from roi_id to original annotation
        annotation_map = {ann["roi_id"]: ann for ann in original_annotations}
        
        filtered_annotations = []
        
        for img in filtered_images:
            roi_id = img["id"]
            if roi_id in annotation_map:
                # Use the original annotation structure but update with any changes
                filtered_ann = annotation_map[roi_id].copy()
                # Update with any fields that might have changed
                for key, value in img.items():
                    if key not in ["id", "_embedding"]:  # Skip internal fields
                        filtered_ann[key] = value
                filtered_annotations.append(filtered_ann)
        
        return filtered_annotations
    
    def get_filter_info(self) -> Dict[str, Any]:
        """Get information about the adapted filter."""
        base_info = self.clustering_filter.get_filter_info()
        base_info["adapter_type"] = "CropClusteringAdapter"
        return base_info


class ClassificationRebalanceFilter(BaseFilter):
    """
    Filter to rebalance a skewed dataset for image classification by undersampling majority classes.
    Input: List[dict] representing annotations (each dict must have 'class_id' or 'class_name')
    Output: List[dict] filtered annotations with balanced class distribution
    """
    def __init__(self, class_key: str = "class_id", random_seed: Optional[int] = 41, method: str = "mean",exclude_extremes: bool = True):
        """
        Args:
            class_key: The key in annotation dicts to use for class label (e.g., 'class_id' or 'class_name')
            random_seed: Optional random seed for reproducibility
        """
        self.class_key = class_key
        self.random_seed = random_seed
        self.method = method
        self.exclude_extremes = exclude_extremes
        
    def _get_mean_count(self, annotations: list[dict]) -> int:
        class_to_anns = defaultdict(list)
        for ann in annotations:
            class_to_anns[ann[self.class_key]].append(ann)
        class_counts = [len(anns) for anns in class_to_anns.values()]
        if self.exclude_extremes and len(class_counts) >= 3:
            class_counts = sorted(class_counts)
            class_counts = class_counts[1:-1]  # Exclude smallest and largest
        return int(round(sum(class_counts) / len(class_counts))) 
    
    def _get_min_count(self, annotations: list[dict]) -> int:
        class_to_anns = defaultdict(list)
        for ann in annotations:
            class_to_anns[ann[self.class_key]].append(ann)
        class_counts = [len(anns) for anns in class_to_anns.values()]
        if self.exclude_extremes and len(class_counts) >= 3:
            class_counts = sorted(class_counts)
            class_counts = class_counts[1:-1]  # Exclude smallest and largest
        return min(class_counts)

    def __call__(self, annotations: list[dict]) -> list[dict]:
        if not annotations:
            return []
        # Group annotations by class
        if self.method == "mean":
            mean_count = self._get_mean_count(annotations)
        elif self.method == "min":
            mean_count = self._get_min_count(annotations)
        else:
            raise ValueError(f"Invalid method: {self.method}")
        
        # Group annotations by class
        class_to_anns = defaultdict(list)
        for ann in annotations:
            class_to_anns[ann[self.class_key]].append(ann)

        # Undersample each class to mean_count
        balanced_anns = []
        rng = random.Random(self.random_seed)
        for anns in class_to_anns.values():
            if len(anns) > mean_count:
                balanced_anns.extend(rng.sample(anns, mean_count))
            else:
                balanced_anns.extend(anns)
        rng.shuffle(balanced_anns)
        return balanced_anns

    def get_filter_info(self) -> dict:
        return {"filter_type": self.__class__.__name__, "class_key": self.class_key, "random_seed": self.random_seed}


class CropClusteringFilter(ClusteringFilter):
    """
    Specialized ClusteringFilter for crop datasets that handles in-memory crops.
    
    This filter extracts crop images from the dataset and computes embeddings
    directly from the image data instead of trying to load files from disk.
    """
    
    def __init__(self, 
                 crop_dataset: 'PatchDataset',
                 feature_extractor: Optional['FeatureExtractor'] = None,
                 batch_size: int = 64,
                 sampling_strategy: Optional[SamplingStrategy] = None,
                 reduction_factor: float = 0.3):
        """
        Initialize the crop clustering filter.
        
        Args:
            crop_dataset: The PatchDataset instance to filter
            feature_extractor: Feature extractor for computing embeddings
            batch_size: Batch size for processing
            sampling_strategy: Sampling strategy for cluster selection
            reduction_factor: Fraction of crops to keep
        """
        super().__init__(feature_extractor, batch_size, sampling_strategy, reduction_factor)
        self.crop_dataset = crop_dataset
    
    def add_embeddings(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to images by extracting crops and computing features.
        
        Args:
            images: List of image dictionaries with crop info
            
        Returns:
            List of images with embeddings added
        """
        if not images:
            logger.warning("No images found in images")
            return images
        
        # Extract crop images and compute embeddings
        crop_images = []
        for img in tqdm(images,desc="Extracting crops",unit="img"):
            crop_info = img.get("_crop_info")
            if crop_info is not None:
                # Extract the actual crop image
                crop_image = self.crop_dataset._extract_crop(crop_info)
                crop_images.append(crop_image)
            else:
                # Fallback: create a placeholder
                crop_images.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Compute embeddings in batches
        embeddings = self._compute_embeddings_from_images(crop_images)
        
        # Add embeddings to images
        for img, emb in zip(images, embeddings):
            img["_embedding"] = emb
        
        return images
    
    def _compute_embeddings_from_images(self, crop_images: List[np.ndarray]) -> np.ndarray:
        """
        Compute embeddings from crop images.
        
        Args:
            crop_images: List of crop images as numpy arrays
            
        Returns:
            Array of embeddings
        """
        all_embeddings = []
        for i in tqdm(range(0, len(crop_images), self.batch_size),desc="Computing embeddings",unit="batch"):
            batch_images = crop_images[i:i + self.batch_size]
            # Convert to PIL Images for the feature extractor
            batch = torch.stack([torch.from_numpy(img_array).permute(2,0,1) for img_array in batch_images])
            # Get embeddings from the model
            with torch.no_grad():
                batch_embeddings = self.feature_extractor(batch).cpu().numpy()
                #batch_embeddings = outputs.cpu().reshape(len(batch_pil_images), -1).numpy()
            all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)


class FilterDataCfg:
    def __init__(self,data_config_yaml: str,keep_classes:Optional[list[str]]=None,discard_classes:Optional[list[str]]=None):
        self.data_config_yaml = data_config_yaml
        self.yolo_config = load_yaml(data_config_yaml)
        self.keep_classes = keep_classes
        self.discard_classes = discard_classes
        assert (keep_classes is not None) ^ (discard_classes is not None), "Either keep_classes or discard_classes must be provided"
    
    def sample_pos_neg(self,images_paths: list, ratio: float, seed: int = 41):
        """
        Sample positive and negative image paths based on the ratio of empty to non-empty samples.

        Args:
            images_paths (list): List of image paths.
            ratio (float): Ratio defined as num_empty/num_non_empty.
            seed (int, optional): Random seed. Defaults to 41.

        Returns:
            list: Selected image paths.
        """

        # build dataframe
        is_empty = [
            1 - os.path.exists(Path(p).with_suffix(".txt").as_posix().replace("/images/", "/labels/"))
            for p in images_paths
        ]
        data = pd.DataFrame.from_dict(
            {"image_paths": images_paths, "is_empty": is_empty}, orient="columns"
        )
        # get empty and non empty
        num_empty = (data["is_empty"] == 1).sum()
        num_non_empty = len(data) - num_empty
        if num_empty == 0:
            logger.info("contains only positive samples")
        num_sampled_empty = min(int(num_non_empty * ratio), num_empty)
        sampled_empty = data.loc[data["is_empty"] == 1].sample(
            n=num_sampled_empty, random_state=seed
        )
        # concatenate
        sampled_data = pd.concat([sampled_empty, data.loc[data["is_empty"] == 0]])

        logger.info(f"Sampling: pos={num_non_empty} & neg={num_sampled_empty}")

        return sampled_data["image_paths"].to_list()
    
    def get_split_images_paths(self,split:str) -> list[str]:
        root = self.yolo_config["path"]
        if isinstance(self.yolo_config[split], list):
            dirs_images = [os.path.join(root, p) for p in self.yolo_config[split]]
        else:
            dirs_images = [self.yolo_config[split]]

        images_paths = []
        for dir_images in dirs_images:
            images_paths.extend(list(Path(dir_images).glob("*")))
        
        return images_paths
    
    @staticmethod
    def filter_image_paths(image_paths: list[str],label_map:dict,
                            keep_classes:Optional[list[str]]=None,
                            discard_classes:Optional[list[str]]=None) -> list[str]:

        assert (keep_classes is not None) ^ (discard_classes is not None), "Either keep_classes or discard_classes must be provided"
        
        filtered_images_paths = []

        for image_path in image_paths:
            label_path = Path(image_path).with_suffix(".txt").as_posix().replace("/images/", "/labels/")

            if not os.path.exists(label_path):
                filtered_images_paths.append(image_path)
                continue
            
            with open(label_path, "r", encoding="utf-8") as f:
                labels = [line.split(" ")[0] for line in f.read().splitlines()]
                classes = [label_map[int(label)] for label in labels]
                
                # If no labels in the image, skip it
                if len(classes) == 0:
                    filtered_images_paths.append(image_path)
                    continue
                
                should_include = True
                
                # Apply keep_classes filter if specified
                if keep_classes is not None:
                    should_include = any(class_name in keep_classes for class_name in classes)
                
                # Apply discard_classes filter if specified
                if discard_classes is not None and should_include:
                    should_include = not any(class_name in discard_classes for class_name in classes)
                
                if should_include:
                    filtered_images_paths.append(image_path)
                
        return filtered_images_paths

    def filter_data_cfg(self, split: str) -> list[str]:
        images_paths = self.get_split_images_paths(split=split)
        if self.keep_classes is None and self.discard_classes is None:
            return images_paths
        filtered_images_paths = self.filter_image_paths(image_paths=images_paths,
                                                        label_map=self.yolo_config["names"],
                                                        keep_classes=self.keep_classes,
                                                        discard_classes=self.discard_classes)
        if self.keep_classes is not None:
            logger.info(f"Keeping classes for split {split}: {self.keep_classes}")
        if self.discard_classes is not None:
            logger.info(f"Discarding classes for split {split}: {self.discard_classes}")
                
        return filtered_images_paths
    
    def save_images_paths(self,images_paths:list[str],save_path:str):
        pd.Series(images_paths).to_csv(save_path,index=False,header=False)
    
    def get_filtered_data_cfg(self,save_dir:str) -> str:
        root = self.yolo_config["path"]

        # updated cfg
        cfg = {
            "path": root,
            "names": self.yolo_config["names"],
            "nc": self.yolo_config["nc"]
        }
        splits = [t for t in ["train","val","test"] if t in self.yolo_config.keys()]
        for split in splits:
            images_paths = self.filter_data_cfg(split=split)
            save_path_samples = os.path.join(
                save_dir, f"{split}_flatten.txt"
            )
            self.save_images_paths(images_paths=images_paths,save_path=save_path_samples)

            # update cfg
            cfg[split] = os.path.relpath(save_path_samples, start=root)
        
        # save cfg
        save_path_cfg = Path(self.data_config_yaml).with_stem("filtered_data").as_posix()
        save_yaml(cfg,save_path_cfg,mode="w")
        
        return save_path_cfg
            
    def get_data_cfg_paths_for_cl(
        self,
        ratio: float,
        split:str,
        cl_save_dir: str,
        seed: int = 41,
    ):
        """
        Generate and save a YOLO data config YAML for continual learning with sampled images.

        Args:
            ratio (float): Ratio for sampling.
            data_config_yaml (str): Path to YOLO data config YAML.
            cl_save_dir (str): Directory to save sampled images and config.
            seed (int, optional): Random seed. Defaults to 41.

        Returns:
            str: Path to the saved config YAML.
        """

        root = self.yolo_config["path"]

        images_paths = self.filter_data_cfg(split=split)

        sampled_imgs_paths = self.sample_pos_neg(
            images_paths=images_paths,
            ratio=ratio,
            seed=seed,
        )

        # save selected images in txt file
        save_path_samples = os.path.join(
            cl_save_dir, f"{split}_ratio_{ratio}-seed_{seed}.txt"
        )
        self.save_images_paths(images_paths=sampled_imgs_paths,save_path=save_path_samples)

        # updated cfg
        cfg = {
            "path": root,
            "names": self.yolo_config["names"],
            "nc": self.yolo_config["nc"]
        }
        if split == "train":
            cfg["val"] = self.yolo_config["val"]
            cfg["train"] = os.path.relpath(save_path_samples, start=root)
        elif split == "val":
            cfg["val"] = os.path.relpath(save_path_samples, start=root)
            cfg["train"] = self.yolo_config["train"]
        else:
            raise NotImplementedError

        save_path_cfg = Path(save_path_samples).with_suffix(".yaml").as_posix()
        save_yaml(cfg,save_path_cfg,mode="w")

        logger.info(
            f"Saving samples at: {save_path_samples} and data_cfg at {save_path_cfg}",
        )

        return str(save_path_cfg)