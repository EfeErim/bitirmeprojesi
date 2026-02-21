# Dynamic Plant Detection Guide

## How It Works Now

Your pipeline now supports **universal plant detection** with three approaches:

### **1. Dynamic Taxonomy (Enabled by Default in Colab)**

```json
{
  "router": {
    "vlm": {
      "use_dynamic_taxonomy": true,
      "taxonomy_path": "config/plant_taxonomy.json"
    }
  }
}
```

**What happens:**
- Loads 79+ plant types from [plant_taxonomy.json](../config/plant_taxonomy.json)
- Covers: crops (tomato, corn, wheat, rice...), weeds (dandelion, thistle...), ornamentals (rose, tulip...)
- 16 plant parts: leaf, fruit, flower, stem, root, seed, tuber, bulb, pod, grain, branch, trunk, bark, bud, shoot
- GroundingDINO uses **generic prompts** (plant, leaf, crop) for detection
- BioCLIP classifies detected regions against full taxonomy
- Open-set rejection: returns `'unknown'` if confidence is too low

**Advantages:**
✅ Works for most common plants without config changes  
✅ Single config covers agriculture, horticulture, weed detection  
✅ BioCLIP can identify 79+ plant types zero-shot

**Limitations:**
- If plant isn't in taxonomy file, returns `'unknown'` (but still detects it)
- Takes ~2-3s for first inference (loads taxonomy + models)

---

### **2. Custom Taxonomy (For Specialized Use Cases)**

Create your own taxonomy file:

```json
{
  "crops": ["your_crop_1", "your_crop_2", "..."],
  "parts": ["leaf", "fruit", "custom_part"],
  "common_weeds": ["weed_species_1"],
  "ornamentals": []
}
```

Point config to it:
```json
{
  "vlm": {
    "use_dynamic_taxonomy": true,
    "taxonomy_path": "path/to/your_taxonomy.json"
  }
}
```

**Use this when:**
- Specialized crops (rare species, exotic plants)
- Domain-specific (only brassicas, only citrus, etc.)
- Custom workflow needs (include diseases in taxonomy)

---

### **3. Specific Crops Only (Original Behavior)**

```json
{
  "vlm": {
    "use_dynamic_taxonomy": false,
    "crop_labels": ["tomato", "potato", "grape", "strawberry"],
    "part_labels": ["leaf", "fruit", "stem"]
  }
}
```

**What happens:**
- GroundingDINO uses **specific prompts** (tomato, potato, grape, strawberry)
- Better detection precision when you know exactly what plants to expect
- Lower false positives

**Use this when:**
- Fixed set of crops in controlled environment (greenhouse, specific farm)
- Need maximum precision for known crop types
- Want faster inference (fewer labels = faster BioCLIP)

---

## Architecture Flow

```
Image Upload
    ↓
GroundingDINO Detection
├─ Dynamic taxonomy? → Generic prompts: "plant. leaf. crop. flower. fruit."
└─ Specific crops?    → Specific prompts: "tomato. potato. grape. strawberry."
    ↓
Best Detection Selected
    ↓
ROI Extracted (bbox + 8% padding)
    ↓
BioCLIP-2 Classification
├─ Prompt ensemble per label (4 templates × N labels)
├─ Max-pooling across templates
├─ Softmax over [known_labels + unknown]
└─ Open-set rejection if confidence < 0.55 or margin < 0.10
    ↓
Result: {crop: "corn", part: "leaf", confidence: 0.87}
```

---

## Expanding the Taxonomy

To add more plants, edit [config/plant_taxonomy.json](../config/plant_taxonomy.json):

```json
{
  "crops": [
    "tomato", "potato", 
    "NEW_CROP_HERE",  ← Add your crops
    "avocado", "papaya"
  ],
  "parts": [
    "leaf", "fruit",
    "NEW_PART_HERE"  ← Add custom parts
  ]
}
```

**Recommendations:**
- Keep total labels < 200 for fast inference
- Use common names (BioCLIP recognizes both scientific + common)
- Test with real images after adding new labels

---

## Performance Tuning

### For Maximum Coverage (Current Default)
```json
{
  "use_dynamic_taxonomy": true,
  "confidence_threshold": 0.3,
  "open_set_min_confidence": 0.50
}
```
→ Detects most plants, more `'unknown'` results

### For Maximum Precision
```json
{
  "use_dynamic_taxonomy": false,
  "crop_labels": ["very", "specific", "crops"],
  "confidence_threshold": 0.7,
  "open_set_min_confidence": 0.65
}
```
→ Fewer detections, higher confidence when it does detect

### For Speed
- Reduce taxonomy size (fewer labels = faster BioCLIP)
- Use specific crops mode with <10 labels
- Lower `max_detections` (default: 10)

---

## Testing Your Setup

```python
# In Colab:
%run scripts/colab_vlm_quick_test.py

# Upload any plant image - it will now work with:
# - Common crops (tomato, corn, wheat, rice, soybean...)
# - Weeds (dandelion, thistle, clover...)
# - Ornamentals (rose, tulip, daisy...)
# - And returns 'unknown' for anything else
```

---

## BioCLIP-2 Full Capability

BioCLIP-2 is trained on **450k+ biodiversity concepts** from iNaturalist.  

**Why not use all 450k?**
- Inference would take 60+ seconds per image
- Most labels are animals, insects, fungi (not relevant)
- Prompt-ensemble with 450k labels = out of memory

**Future Enhancement:**
Could implement hierarchical classification:
1. First pass: "Is this Plantae kingdom?" → Yes
2. Second pass: "Which family?" → Solanaceae
3. Third pass: "Which genus?" → Solanum
4. Final pass: "Which species?" → tomato

This would enable true open-vocabulary plant ID without predefined lists.

---

## Comparison with Other Approaches

| Approach | Coverage | Speed | Precision | Flexibility |
|----------|----------|-------|-----------|-------------|
| **Dynamic Taxonomy (Current)** | High (79+) | Medium | Medium-High | High |
| Specific Crops | Low (4-10) | Fast | High | Low |
| Hierarchical BioCLIP | Very High (450k) | Slow | Medium | Very High |
| Generic Detection Only | Very High | Very Fast | Low | Very High |

**Recommendation:** Use dynamic taxonomy (current default) unless you have a specific reason to change.

---

## Troubleshooting

**Issue: Getting `'unknown'` for common plants**  
→ Check if plant is in [plant_taxonomy.json](../config/plant_taxonomy.json)  
→ Lower `open_set_min_confidence` to 0.45  
→ Add plant to taxonomy file

**Issue: Wrong classifications**  
→ Check confidence scores - are they close (e.g., 0.52 vs 0.48)?  
→ Increase `open_set_margin` to 0.15 for stricter rejection  
→ Use specific crop mode for better precision

**Issue: No detections (bbox [0, 0, 100, 100])**  
→ Lower `confidence_threshold` to 0.2  
→ Verify image has clear plant features  
→ Check GroundingDINO output in logs: `det.get('grounding_label')`

---

## Next Steps

1. **Test with your images**: Upload various plant types and check results
2. **Tune thresholds**: Adjust confidence/margin based on your precision needs
3. **Expand taxonomy**: Add specific plants for your domain
4. **Monitor performance**: Check `processing_time_ms` in results

The system is now **production-ready** for general-purpose plant detection! 🎉
