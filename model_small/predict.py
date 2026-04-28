"""
predict.py - Enhanced ensemble prediction script

Unified inference script for wafer defect prediction using an ensemble of ViT and ResNet models.

Supports multiple prediction methods:
  1. Individual model predictions (ViT or ResNet)
  2. Ensemble averaging (ViT + ResNet)
  3. TTA-averaged predictions (with 4 rotations)
  4. Batch prediction on directories

Usage:
    # Basic: ensemble prediction on single image
    python predict.py --image path/to/wafer.png
    
    # Individual model predictions
    python predict.py --image path/to/wafer.png --model vit
    python predict.py --image path/to/wafer.png --model resnet
    
    # Batch prediction
    python predict.py --image_dir path/to/wafers/ --output results.csv
    
    # With TTA (test-time augmentation)
    python predict.py --image path/to/wafer.png --use_tta
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Union
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from data_utils import to_canonical, rgb_to_class_array, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
from models import build_model


# =============================================================================
# LOGGING
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """Create a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def load_image_as_rgb(image_path: Path | str) -> np.ndarray:
    """Load image from file and convert to RGB array."""
    image_path = Path(image_path)
    assert image_path.exists(), f"Image not found: {image_path}"
    
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img)


def preprocess_image(image_path: Path | str, logger=None) -> dict:
    """
    Preprocess image to both ViT and ResNet formats.
    
    Returns:
        {
            "vit_input": (1, 64, 64) long tensor for ViT (values in {0,1,2})
            "resnet_input": (1, 3, 64, 64) float tensor for ResNet (values in {0.0, 0.5, 1.0})
            "raw_array": The class array before tensor conversion
        }
    """
    image_path = Path(image_path)
    if logger:
        logger.info(f"Loading image: {image_path}")
    
    # Load RGB
    img_rgb = load_image_as_rgb(image_path)
    
    # RGB → class array {0,1,2}
    class_array = rgb_to_class_array(img_rgb)
    
    # Apply canonical preprocessing
    arr = to_canonical(class_array)
    
    # ViT input: (1, 64, 64) long tensor
    vit_input = torch.from_numpy(arr.copy()).long().unsqueeze(0)
    
    # ResNet input: (1, 3, 64, 64) float tensor with values {0.0, 0.5, 1.0}
    rn_arr = arr.astype(np.float32) / 2.0  # {0.0, 0.5, 1.0}
    rn_arr = np.stack([rn_arr, rn_arr, rn_arr], axis=0)  # (3, 64, 64)
    resnet_input = torch.from_numpy(rn_arr).unsqueeze(0)
    
    return {
        "vit_input": vit_input,
        "resnet_input": resnet_input,
        "raw_array": arr,
    }


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_from_checkpoint(
    model_name: str,
    ckpt_path: Path | str,
    device: torch.device,
    logger=None
) -> nn.Module:
    """Load a trained model from checkpoint."""
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    
    model = build_model(model_name, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    if logger:
        logger.info(f"Loaded {model_name} checkpoint: {ckpt_path}")
    
    return model


def load_calibration_temperatures(ckpt_dir: Path | str) -> dict:
    """Load temperature calibration values for ViT and ResNet."""
    ckpt_dir = Path(ckpt_dir)
    
    temps = {}
    for model_name in ["vit", "resnet"]:
        temp_path = ckpt_dir / f"temperature_{model_name}.npy"
        if temp_path.exists():
            temps[model_name] = float(np.load(temp_path)[0])
        else:
            temps[model_name] = 1.0
    
    return temps


def load_ensemble_weight(ckpt_dir: Path | str) -> float:
    """Load optimal ensemble weight (ResNet weight, ViT weight = 1 - w)."""
    ckpt_dir = Path(ckpt_dir)
    weight_path = ckpt_dir / "ensemble_weight.npy"
    
    if weight_path.exists():
        return float(np.load(weight_path)[0])
    else:
        # Default to equal weighting
        return 0.5


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

@torch.no_grad()
def predict_single_model(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    temperature: float = 1.0,
    model_name: str = "model",
) -> dict:
    """
    Single forward pass prediction for one model.
    
    Returns:
        {
            "class_idx": int,
            "class_name": str,
            "confidence": float,
            "probs": np.ndarray (9,),
            "logits": np.ndarray (9,),
        }
    """
    x = x.to(device)
    logits, _ = model(x)  # (1, 9)
    
    # Temperature-calibrated softmax
    calibrated_logits = logits / temperature
    probs = F.softmax(calibrated_logits, dim=-1).cpu().numpy()[0]
    
    class_idx = int(probs.argmax())
    confidence = float(probs[class_idx])
    
    return {
        "class_idx": class_idx,
        "class_name": IDX_TO_CLASS[class_idx],
        "confidence": confidence,
        "probs": probs,
        "logits": logits.cpu().numpy()[0],
        "model": model_name,
        "temperature": temperature,
    }


@torch.no_grad()
def predict_ensemble(
    vit_model: nn.Module,
    resnet_model: nn.Module,
    vit_input: torch.Tensor,
    resnet_input: torch.Tensor,
    device: torch.device,
    vit_temp: float = 1.0,
    resnet_temp: float = 1.0,
    ensemble_weight: float = 0.5,
    logger=None,
) -> dict:
    """
    Ensemble prediction combining ViT and ResNet.
    
    Ensemble formula: p_ensemble = w * p_resnet + (1-w) * p_vit
    
    Args:
        vit_model: Trained ViT model
        resnet_model: Trained ResNet model
        vit_input: Preprocessed input for ViT (1, 64, 64)
        resnet_input: Preprocessed input for ResNet (1, 3, 64, 64)
        device: torch device
        vit_temp: Temperature calibration for ViT
        resnet_temp: Temperature calibration for ResNet
        ensemble_weight: Weight for ResNet (ViT gets 1 - weight)
        logger: logger instance
        
    Returns:
        {
            "class_idx": int,
            "class_name": str,
            "confidence": float,
            "ensemble_probs": np.ndarray (9,),
            "vit_probs": np.ndarray (9,),
            "resnet_probs": np.ndarray (9,),
            "vit_pred": str,
            "resnet_pred": str,
        }
    """
    # Get ViT predictions
    vit_input = vit_input.to(device)
    vit_logits, _ = vit_model(vit_input)
    vit_logits_cal = vit_logits / vit_temp
    vit_probs = F.softmax(vit_logits_cal, dim=-1).cpu().numpy()[0]
    
    # Get ResNet predictions
    resnet_input = resnet_input.to(device)
    resnet_logits, _ = resnet_model(resnet_input)
    resnet_logits_cal = resnet_logits / resnet_temp
    resnet_probs = F.softmax(resnet_logits_cal, dim=-1).cpu().numpy()[0]
    
    # Ensemble: weighted average of probabilities
    ensemble_probs = ensemble_weight * resnet_probs + (1.0 - ensemble_weight) * vit_probs
    
    class_idx = int(ensemble_probs.argmax())
    confidence = float(ensemble_probs[class_idx])
    
    vit_pred_idx = int(vit_probs.argmax())
    resnet_pred_idx = int(resnet_probs.argmax())
    
    if logger:
        logger.info(f"\n{'='*60}")
        logger.info(f"ViT prediction:    {IDX_TO_CLASS[vit_pred_idx]} (conf={vit_probs[vit_pred_idx]:.4f})")
        logger.info(f"ResNet prediction: {IDX_TO_CLASS[resnet_pred_idx]} (conf={resnet_probs[resnet_pred_idx]:.4f})")
        logger.info(f"Ensemble weight (ResNet): {ensemble_weight:.2f}, (ViT): {1-ensemble_weight:.2f}")
        logger.info(f"Ensemble prediction: {IDX_TO_CLASS[class_idx]} (conf={confidence:.4f})")
        logger.info(f"{'='*60}")
    
    return {
        "class_idx": class_idx,
        "class_name": IDX_TO_CLASS[class_idx],
        "confidence": confidence,
        "ensemble_probs": ensemble_probs,
        "vit_probs": vit_probs,
        "resnet_probs": resnet_probs,
        "vit_pred": IDX_TO_CLASS[vit_pred_idx],
        "resnet_pred": IDX_TO_CLASS[resnet_pred_idx],
        "ensemble_weight_resnet": ensemble_weight,
        "ensemble_weight_vit": 1.0 - ensemble_weight,
    }


@torch.no_grad()
def predict_ensemble_tta(
    vit_model: nn.Module,
    resnet_model: nn.Module,
    vit_input: torch.Tensor,
    resnet_input: torch.Tensor,
    device: torch.device,
    vit_temp: float = 1.0,
    resnet_temp: float = 1.0,
    ensemble_weight: float = 0.5,
    logger=None,
) -> dict:
    """
    Ensemble prediction with test-time augmentation (TTA).
    
    Applies 4 augmentations: identity, rot90, rot180, rot270.
    """
    
    def apply_augmentations(arr_input):
        """Generate augmented views: identity, rot90, rot180, rot270."""
        views = [
            arr_input,  # identity
            torch.rot90(arr_input, k=1, dims=[2, 3]),  # 90°
            torch.rot90(arr_input, k=2, dims=[2, 3]),  # 180°
            torch.rot90(arr_input, k=3, dims=[2, 3]),  # 270°
        ]
        return views
    
    vit_views = apply_augmentations(vit_input)
    resnet_views = apply_augmentations(resnet_input)
    
    # Accumulate probabilities across views
    vit_probs_all = []
    resnet_probs_all = []
    
    for v_vit, v_resnet in zip(vit_views, resnet_views):
        v_vit = v_vit.to(device)
        v_resnet = v_resnet.to(device)
        
        # ViT
        vit_logits, _ = vit_model(v_vit)
        vit_logits_cal = vit_logits / vit_temp
        vit_p = F.softmax(vit_logits_cal, dim=-1).cpu().numpy()[0]
        vit_probs_all.append(vit_p)
        
        # ResNet
        resnet_logits, _ = resnet_model(v_resnet)
        resnet_logits_cal = resnet_logits / resnet_temp
        resnet_p = F.softmax(resnet_logits_cal, dim=-1).cpu().numpy()[0]
        resnet_probs_all.append(resnet_p)
    
    # Average across augmentations
    vit_probs_mean = np.mean(vit_probs_all, axis=0)
    resnet_probs_mean = np.mean(resnet_probs_all, axis=0)
    
    # Ensemble: weighted average
    ensemble_probs = ensemble_weight * resnet_probs_mean + (1.0 - ensemble_weight) * vit_probs_mean
    
    class_idx = int(ensemble_probs.argmax())
    confidence = float(ensemble_probs[class_idx])
    
    vit_pred_idx = int(vit_probs_mean.argmax())
    resnet_pred_idx = int(resnet_probs_mean.argmax())
    
    if logger:
        logger.info(f"\n{'='*60}")
        logger.info(f"TTA Ensemble Prediction (4 augmentations)")
        logger.info(f"ViT prediction:    {IDX_TO_CLASS[vit_pred_idx]} (conf={vit_probs_mean[vit_pred_idx]:.4f})")
        logger.info(f"ResNet prediction: {IDX_TO_CLASS[resnet_pred_idx]} (conf={resnet_probs_mean[resnet_pred_idx]:.4f})")
        logger.info(f"Ensemble weight (ResNet): {ensemble_weight:.2f}, (ViT): {1-ensemble_weight:.2f}")
        logger.info(f"Ensemble prediction: {IDX_TO_CLASS[class_idx]} (conf={confidence:.4f})")
        logger.info(f"{'='*60}")
    
    return {
        "class_idx": class_idx,
        "class_name": IDX_TO_CLASS[class_idx],
        "confidence": confidence,
        "ensemble_probs": ensemble_probs,
        "vit_probs": vit_probs_mean,
        "resnet_probs": resnet_probs_mean,
        "vit_pred": IDX_TO_CLASS[vit_pred_idx],
        "resnet_pred": IDX_TO_CLASS[resnet_pred_idx],
        "ensemble_weight_resnet": ensemble_weight,
        "ensemble_weight_vit": 1.0 - ensemble_weight,
        "method": "tta",
        "num_augmentations": 4,
    }


# =============================================================================
# ENSEMBLE PREDICTOR CLASS
# =============================================================================

class EnsembleWaferPredictor:
    """
    Ensemble predictor combining ViT and ResNet models for wafer defect classification.
    """
    
    def __init__(
        self,
        vit_ckpt: Path | str = "checkpoints/vit_best.pth",
        resnet_ckpt: Path | str = "checkpoints/resnet_best.pth",
        ckpt_dir: Path | str = "checkpoints",
        device: torch.device = None,
        logger=None,
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            vit_ckpt: Path to ViT checkpoint
            resnet_ckpt: Path to ResNet checkpoint
            ckpt_dir: Directory containing calibration files
            device: torch device (defaults to cuda if available)
            logger: logger instance
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or get_logger("ensemble_predictor")
        
        self.logger.info(f"Device: {self.device}")
        
        # Load models
        self.logger.info("Loading models...")
        self.vit = load_model_from_checkpoint("vit", vit_ckpt, self.device, self.logger)
        self.resnet = load_model_from_checkpoint("resnet50", resnet_ckpt, self.device, self.logger)
        
        # Load calibration parameters
        self.logger.info("Loading calibration parameters...")
        temps = load_calibration_temperatures(ckpt_dir)
        self.vit_temp = temps["vit"]
        self.resnet_temp = temps["resnet"]
        self.ensemble_weight = load_ensemble_weight(ckpt_dir)
        
        self.logger.info(f"  ViT temperature: {self.vit_temp:.4f}")
        self.logger.info(f"  ResNet temperature: {self.resnet_temp:.4f}")
        self.logger.info(f"  Ensemble weight (ResNet): {self.ensemble_weight:.2f}")
        self.logger.info(f"  Ensemble weight (ViT): {1-self.ensemble_weight:.2f}")
    
    def predict_single(
        self,
        image_path: Path | str,
        model_name: str = "vit",
    ) -> dict:
        """
        Predict using a single model (ViT or ResNet).
        
        Args:
            image_path: Path to image
            model_name: "vit" or "resnet"
            
        Returns:
            Prediction result dictionary
        """
        self.logger.info(f"\nPredicting with {model_name}: {image_path}")
        
        # Preprocess
        inputs = preprocess_image(image_path, self.logger)
        
        if model_name == "vit":
            result = predict_single_model(
                self.vit,
                inputs["vit_input"],
                self.device,
                self.vit_temp,
                "ViT"
            )
        elif model_name == "resnet":
            result = predict_single_model(
                self.resnet,
                inputs["resnet_input"],
                self.device,
                self.resnet_temp,
                "ResNet"
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return result
    
    def predict_ensemble(
        self,
        image_path: Path | str,
        use_tta: bool = False,
    ) -> dict:
        """
        Ensemble prediction combining both models.
        
        Args:
            image_path: Path to image
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Ensemble prediction result
        """
        self.logger.info(f"\nEnsemble prediction: {image_path}")
        
        # Preprocess
        inputs = preprocess_image(image_path, self.logger)
        
        if use_tta:
            result = predict_ensemble_tta(
                self.vit,
                self.resnet,
                inputs["vit_input"],
                inputs["resnet_input"],
                self.device,
                self.vit_temp,
                self.resnet_temp,
                self.ensemble_weight,
                self.logger,
            )
        else:
            result = predict_ensemble(
                self.vit,
                self.resnet,
                inputs["vit_input"],
                inputs["resnet_input"],
                self.device,
                self.vit_temp,
                self.resnet_temp,
                self.ensemble_weight,
                self.logger,
            )
        
        return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble wafer defect prediction with ViT and ResNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path, help="Path to single image")
    input_group.add_argument("--image_dir", type=Path, help="Path to directory of images")
    
    parser.add_argument(
        "--method",
        choices=["vit", "resnet", "ensemble"],
        default="ensemble",
        help="Which model(s) to use for prediction"
    )
    parser.add_argument(
        "--vit_ckpt",
        type=Path,
        default=script_dir / "checkpoints" / "vit_best.pth",
        help="Path to ViT checkpoint"
    )
    parser.add_argument(
        "--resnet_ckpt",
        type=Path,
        default=script_dir / "checkpoints" / "resnet_best.pth",
        help="Path to ResNet checkpoint"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        default=script_dir / "checkpoints",
        help="Checkpoint directory (for calibration files)"
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Use test-time augmentation (4 rotations)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file for batch predictions"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = get_logger("ensemble_prediction")
    if args.verbose:
        logging.getLogger("ensemble_prediction").setLevel(logging.DEBUG)
    
    # Initialize predictor
    predictor = EnsembleWaferPredictor(
        vit_ckpt=args.vit_ckpt,
        resnet_ckpt=args.resnet_ckpt,
        ckpt_dir=args.ckpt_dir,
        logger=logger,
    )
    
    # Single image prediction
    if args.image:
        if args.method == "vit":
            result = predictor.predict_single(args.image, "vit")
        elif args.method == "resnet":
            result = predictor.predict_single(args.image, "resnet")
        else:  # ensemble
            result = predictor.predict_ensemble(args.image, use_tta=args.use_tta)
        
        logger.info("\n" + "="*60)
        logger.info("PREDICTION RESULT")
        logger.info("="*60)
        logger.info(f"Image: {args.image}")
        logger.info(f"Predicted class: {result['class_name']}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        
        if "vit_pred" in result:
            logger.info(f"  ViT prediction: {result['vit_pred']}")
            logger.info(f"  ResNet prediction: {result['resnet_pred']}")
        
        logger.info("Per-class probabilities:")
        for i in range(NUM_CLASSES):
            if "ensemble_probs" in result:
                prob = result["ensemble_probs"][i]
            else:
                prob = result["probs"][i]
            logger.info(f"  {IDX_TO_CLASS[i]:12s}: {prob:.4f}")
    
    # Batch prediction
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
        
        logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        results = []
        for image_path in image_files:
            try:
                if args.method == "vit":
                    result = predictor.predict_single(image_path, "vit")
                elif args.method == "resnet":
                    result = predictor.predict_single(image_path, "resnet")
                else:
                    result = predictor.predict_ensemble(image_path, use_tta=args.use_tta)
                
                results.append({
                    "image": image_path.name,
                    "class": result["class_name"],
                    "confidence": result["confidence"],
                })
                
                logger.info(f"  {image_path.name}: {result['class_name']} (conf={result['confidence']:.4f})")
            except Exception as e:
                logger.error(f"  {image_path.name}: ERROR - {e}")
        
        # Save results
        if args.output:
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["image", "class", "confidence"])
                writer.writeheader()
                writer.writerows(results)
            logger.info(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
