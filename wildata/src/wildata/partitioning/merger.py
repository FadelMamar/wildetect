"""
COCO Dataset Merger

This module provides a class for merging multiple COCO annotation dicts into a single unified dataset.
It harmonizes categories, remaps image and annotation IDs, and fuses annotations for downstream partitioning.
"""

import collections
import copy
from typing import Any, Dict, List, Optional, Tuple


# TODO: manual tests
class COCODatasetMerger:
    """
    Merges multiple COCO annotation dicts into a unified dataset.
    - Harmonizes categories (by name, case-insensitive)
    - Remaps image and annotation IDs to avoid collisions
    - Fuses annotations and metadata
    """

    def __init__(self, datasets: List[Dict[str, Any]]):
        """
        Args:
            datasets: List of COCO annotation dicts (already loaded as Python dicts)
        """
        self.datasets = datasets
        self.unified_categories: List[Dict[str, Any]] = []
        self.category_name_to_id: Dict[Tuple[str, str], int] = {}
        self.category_remap: List[Dict[int, int]] = []  # Per-dataset old_id -> new_id
        self.image_remap: List[Dict[int, int]] = []  # Per-dataset old_id -> new_id
        self.next_category_id = 1
        self.next_image_id = 1
        self.next_annotation_id = 1
        self.skipped_image_ids: List[set] = []  # Per-dataset set of skipped image ids

    def merge(self) -> Dict[str, Any]:
        """
        Merge all datasets into a single unified COCO annotation dict.
        Returns:
            Unified COCO annotation dict.
        """
        self._harmonize_categories()
        images, image_id_maps = self._merge_images()
        annotations = self._merge_annotations(image_id_maps)
        info, licenses = self._merge_metadata()

        merged: Dict[str, Any] = {
            "images": images,
            "annotations": annotations,
            "categories": self.unified_categories,
        }
        if info is not None:
            merged["info"] = info
        if licenses is not None:
            merged["licenses"] = licenses
        return merged

    def _harmonize_categories(self):
        """
        Build a unified category list and create per-dataset category remapping.
        """
        name_to_cat = collections.OrderedDict()
        for dataset in self.datasets:
            for cat in dataset.get("categories", []):
                key = (cat["name"].strip().lower(), cat.get("supercategory", ""))
                if key not in name_to_cat:
                    new_cat = {
                        "id": self.next_category_id,
                        "name": cat["name"],
                        "supercategory": cat.get("supercategory", ""),
                    }
                    name_to_cat[key] = new_cat
                    self.category_name_to_id[key] = self.next_category_id
                    self.next_category_id += 1
        self.unified_categories = list(name_to_cat.values())

        # Build per-dataset category remapping
        self.category_remap = []
        for dataset in self.datasets:
            remap = {}
            for cat in dataset.get("categories", []):
                key = (cat["name"].strip().lower(), cat.get("supercategory", ""))
                remap[cat["id"]] = self.category_name_to_id[key]
            self.category_remap.append(remap)

    def _merge_images(self) -> Tuple[List[Dict[str, Any]], List[Dict[int, int]]]:
        """
        Merge images from all datasets, remapping IDs to avoid collisions.
        Returns:
            (merged_images, list of per-dataset old_id -> new_id mappings)
        """
        merged_images = []
        image_id_maps = []
        self.skipped_image_ids = []
        seen_paths = set()
        for idx, dataset in enumerate(self.datasets):
            id_map = {}
            skipped = set()
            for img in dataset.get("images", []):
                # Use absolute file path as unique key
                img_path = img.get("file_name")
                if img_path in seen_paths:
                    # Skip duplicate images
                    skipped.add(img["id"])
                    continue
                seen_paths.add(img_path)
                new_img = copy.deepcopy(img)
                new_img_id = self.next_image_id
                id_map[img["id"]] = new_img_id
                new_img["id"] = new_img_id
                self.next_image_id += 1
                merged_images.append(new_img)
            image_id_maps.append(id_map)
            self.skipped_image_ids.append(skipped)
        self.image_remap = image_id_maps
        return merged_images, image_id_maps

    def _merge_annotations(
        self, image_id_maps: List[Dict[int, int]]
    ) -> List[Dict[str, Any]]:
        """
        Merge annotations, remapping image_id and category_id, and assigning new annotation IDs.
        Skip annotations referencing skipped images.
        """
        merged_annotations = []
        for idx, dataset in enumerate(self.datasets):
            cat_remap = self.category_remap[idx]
            img_remap = image_id_maps[idx]
            skipped_imgs = self.skipped_image_ids[idx]
            for ann in dataset.get("annotations", []):
                if ann["image_id"] in skipped_imgs:
                    # Optionally log or warn here
                    continue  # Skip annotation referencing a skipped image
                new_ann = copy.deepcopy(ann)
                new_ann["id"] = self.next_annotation_id
                self.next_annotation_id += 1
                # Remap image_id and category_id
                new_ann["image_id"] = img_remap[ann["image_id"]]
                new_ann["category_id"] = cat_remap[ann["category_id"]]
                merged_annotations.append(new_ann)
        return merged_annotations

    def _merge_metadata(
        self
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Merge info and licenses fields (optional, can be taken from first dataset or merged as a list).
        """
        info = self.datasets[0].get("info") if self.datasets else None
        licenses = self.datasets[0].get("licenses") if self.datasets else None
        return info, licenses
