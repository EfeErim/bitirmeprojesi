#!/usr/bin/env python3
"""
Optimize OOD and OE datasets:
- Detect and report duplicates
- Analyze slice distribution
- Archive consolidation opportunities
- Provide optimization recommendations
"""

import json
import hashlib
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file for duplicate detection."""
    hash_obj = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def analyze_ood_dataset(ood_root: Path) -> Dict:
    """Analyze OOD dataset structure and find duplicates."""
    analysis = {
        "root": str(ood_root),
        "folders_analyzed": [],
        "total_files": 0,
        "duplicates_found": 0,
        "hash_to_files": defaultdict(list),
        "recommendations": []
    }
    
    final_dir = ood_root / "final"
    if not final_dir.exists():
        return analysis
    
    # Scan all images in final directory
    for image_file in final_dir.rglob("*.jpg"):
        analysis["total_files"] += 1
        try:
            file_hash = compute_file_hash(image_file)
            analysis["hash_to_files"][file_hash].append(str(image_file))
        except Exception as e:
            print(f"Error hashing {image_file}: {e}")
    
    # Find duplicates
    for file_hash, files in analysis["hash_to_files"].items():
        if len(files) > 1:
            analysis["duplicates_found"] += len(files) - 1
            analysis["recommendations"].append({
                "type": "duplicate_files",
                "hash": file_hash,
                "files": files,
                "keep": files[0],
                "remove": files[1:]
            })
    
    # Analyze folder structure
    folders = {}
    for folder in final_dir.iterdir():
        if folder.is_dir():
            image_count = len(list(folder.rglob("*.jpg")))
            subfolders = defaultdict(int)
            for subfolder in folder.iterdir():
                if subfolder.is_dir():
                    subfolders[subfolder.name] = len(list(subfolder.rglob("*.jpg")))
            folders[folder.name] = {
                "total_images": image_count,
                "subfolders": dict(subfolders)
            }
    
    analysis["folders_analyzed"] = folders
    
    # Archive consolidation
    archive_files = list(ood_root.glob("*.zip"))
    if archive_files:
        analysis["recommendations"].append({
            "type": "archive_consolidation",
            "archives": [str(f) for f in archive_files],
            "note": "Consider consolidating extracted archives after validation"
        })
    
    return analysis

def analyze_oe_dataset(oe_root: Path) -> Dict:
    """Analyze OE dataset structure and find duplicates."""
    analysis = {
        "root": str(oe_root),
        "folders_analyzed": [],
        "total_files": 0,
        "duplicates_found": 0,
        "hash_to_files": defaultdict(list),
        "recommendations": [],
        "slice_distribution": {}
    }
    
    # Scan all OE folders
    for target_folder in oe_root.iterdir():
        if not target_folder.is_dir() or target_folder.name.startswith("_"):
            continue
        
        image_count = 0
        slices = defaultdict(int)
        
        for image_file in target_folder.rglob("*.jpg"):
            analysis["total_files"] += 1
            image_count += 1
            
            # Extract slice name from path (parent directory name)
            try:
                slice_name = image_file.parent.name
                slices[slice_name] += 1
            except:
                pass
            
            # Check for duplicates
            try:
                file_hash = compute_file_hash(image_file)
                analysis["hash_to_files"][file_hash].append(str(image_file))
            except Exception as e:
                print(f"Error hashing {image_file}: {e}")
        
        analysis["slice_distribution"][target_folder.name] = {
            "total_images": image_count,
            "slices": dict(slices)
        }
    
    # Find duplicates
    for file_hash, files in analysis["hash_to_files"].items():
        if len(files) > 1:
            analysis["duplicates_found"] += len(files) - 1
            analysis["recommendations"].append({
                "type": "duplicate_files",
                "hash": file_hash,
                "files": files,
                "keep": files[0],
                "remove": files[1:]
            })
    
    # Check slice imbalance
    for target_name, data in analysis["slice_distribution"].items():
        slices = data["slices"]
        if len(slices) > 1:
            slice_counts = list(slices.values())
            max_count = max(slice_counts)
            min_count = min(slice_counts)
            if max_count > min_count * 2:
                analysis["recommendations"].append({
                    "type": "slice_imbalance",
                    "target": target_name,
                    "slices": slices,
                    "note": f"Imbalanced slices: {max_count / min_count:.1f}x spread"
                })
    
    return analysis

def generate_optimization_report(ood_analysis: Dict, oe_analysis: Dict) -> Dict:
    """Generate comprehensive optimization report."""
    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "ood_analysis": ood_analysis,
        "oe_analysis": oe_analysis,
        "summary": {
            "ood_duplicates": ood_analysis["duplicates_found"],
            "oe_duplicates": oe_analysis["duplicates_found"],
            "total_ood_files": ood_analysis["total_files"],
            "total_oe_files": oe_analysis["total_files"],
            "optimization_opportunities": len(ood_analysis["recommendations"]) + len(oe_analysis["recommendations"])
        },
        "action_plan": []
    }
    
    # OOD recommendations
    if ood_analysis["duplicates_found"] > 0:
        report["action_plan"].append({
            "priority": "HIGH",
            "action": "Remove OOD duplicates",
            "count": ood_analysis["duplicates_found"],
            "estimated_space_saved_mb": ood_analysis["duplicates_found"] * 0.5  # rough estimate
        })
    
    # OE recommendations
    if oe_analysis["duplicates_found"] > 0:
        report["action_plan"].append({
            "priority": "HIGH",
            "action": "Remove OE duplicates",
            "count": oe_analysis["duplicates_found"],
            "estimated_space_saved_mb": oe_analysis["duplicates_found"] * 0.5
        })
    
    # Archive consolidation
    ood_archives = any(r.get("type") == "archive_consolidation" for r in ood_analysis["recommendations"])
    if ood_archives:
        report["action_plan"].append({
            "priority": "MEDIUM",
            "action": "Consolidate/cleanup OOD archives after validation",
            "note": "Archives should be retained for reproducibility but can be archived"
        })
    
    return report

def main():
    root = Path("d:\\bitirme projesi")
    
    print("🔍 Analyzing OOD dataset...")
    ood_analysis = analyze_ood_dataset(root / "data" / "ood_dataset")
    
    print("🔍 Analyzing OE dataset...")
    oe_analysis = analyze_oe_dataset(root / "data" / "oe_dataset")
    
    print("📊 Generating optimization report...")
    report = generate_optimization_report(ood_analysis, oe_analysis)
    
    # Save report
    output_path = root / "outputs" / "ood_oe_optimization_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"\nOOD Dataset:")
    print(f"  Total files: {ood_analysis['total_files']}")
    print(f"  Duplicates found: {ood_analysis['duplicates_found']}")
    print(f"  Folders: {len(ood_analysis['folders_analyzed'])}")
    
    print(f"\nOE Dataset:")
    print(f"  Total files: {oe_analysis['total_files']}")
    print(f"  Duplicates found: {oe_analysis['duplicates_found']}")
    print(f"  Targets: {len(oe_analysis['slice_distribution'])}")
    
    print(f"\n📋 Action Items: {report['summary']['optimization_opportunities']}")
    for item in report["action_plan"]:
        print(f"  [{item['priority']}] {item['action']}")
    
    return report

if __name__ == "__main__":
    main()
