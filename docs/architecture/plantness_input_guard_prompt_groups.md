# Plantness Input Guard Prompt Groups

This note defines the prompt groups for a calibration-free input guard that rejects clearly non-plant images before adapter disease inference.

The goal is not to solve unknown plant disease rejection. Unknown diseases still belong to the adapter-side OOD/readiness path. This guard answers a narrower question:

- does the image contain a plant, crop, or supported plant part that is plausible enough to send to the router or adapter?

Implementation status: `src/pipeline/input_guard.py` implements the prompt groups for router-driven inference. The guard is configured under `inference.input_guard` and is disabled by default. When enabled, `src/pipeline/router_adapter_runtime.py` evaluates it before adapter loading and returns `status="non_plant_rejected"` with an `input_guard` payload when plant evidence is too weak or non-plant evidence dominates.

## Rationale

The repo's adapter backbone is DINOv3, which provides strong visual features but does not by itself make text-labeled `plant` versus `non-plant` decisions. For a calibration-free plantness gate, use the repo's text-conditioned vision surfaces:

- SAM/Grounding-style plant prompts for object presence when the router is available.
- BioCLIP/OpenCLIP prompt scoring for plant-versus-non-plant evidence.

Literature anchors:

- Grounding DINO supports open-set object detection from text prompts, so `plant`-style prompts are appropriate for a coarse object-presence gate: https://huggingface.co/docs/transformers/v4.43.3/en/model_doc/grounding-dino
- BioCLIP is trained for biological images across plants, animals, and fungi, and supports zero-shot/few-shot biological visual tasks: https://www.microsoft.com/en-us/research/publication/bioclip-a-vision-foundation-model-for-the-tree-of-life/
- BioCLIP 2.5 is explicitly exposed as an OpenCLIP zero-shot image-classification model for biological visual tasks: https://huggingface.co/imageomics/bioclip-2.5-vith14
- CLIP-style zero-shot OOD detection is a credible precedent for using text-conditioned known/unknown evidence without training a new classifier: https://ojs.aaai.org/index.php/AAAI/article/view/20610

## Decision Contract

The input guard should return an `input_guard` payload:

```json
{
  "enabled": true,
  "decision": "pass",
  "is_plant_like": true,
  "method": "bioclip_prompt_plantness",
  "plant_score": 0.73,
  "non_plant_score": 0.18,
  "margin": 0.55,
  "reason": ""
}
```

Reject contract:

```json
{
  "enabled": true,
  "decision": "non_plant_rejected",
  "is_plant_like": false,
  "method": "bioclip_prompt_plantness",
  "plant_score": 0.21,
  "non_plant_score": 0.66,
  "margin": -0.45,
  "reason": "non_plant_score exceeded plant_score by configured margin"
}
```

Default heuristic:

- reject if `plant_score < 0.45`
- reject if `non_plant_score - plant_score >= 0.10`
- reject if router/SAM plant prompts produce no candidate and BioCLIP evidence is also weak
- pass if plant evidence is strong enough, even when the crop species or disease is unknown

These thresholds are heuristic defaults, not calibrated guarantees. They can be moved into `inference.input_guard` config, but the first implementation should not require a calibration step.

## Positive Prompt Groups

Use multiple prompt groups rather than one literal `plant` prompt. The guard should aggregate each group internally, then aggregate the positive groups into `plant_score`.

### Generic Plant Presence

These prompts catch broad plant content before crop-specific reasoning.

```text
a plant
a green plant
a living plant
a crop plant
an agricultural plant
a cultivated plant
a plant growing in soil
a close-up photo of a plant
a photo of vegetation
plant material
botanical subject
```

### Plant Parts

These prompts prevent legitimate leaf, fruit, stem, and whole-plant views from being rejected just because the full plant is not visible.

```text
a leaf
plant leaves
a crop leaf
a diseased leaf
a healthy leaf
a fruit on a plant
a crop fruit
a plant stem
a crop stem
a plant shoot
a plant branch
a flower on a plant
a root or tuber from a plant
```

### Crop And Farm Plant Context

These prompts help with field images that contain visible crops. They should not be enough by themselves when the image is only landscape or machinery.

```text
a crop in a field
crop rows with visible plants
an agricultural crop
a farm crop plant
vegetable crop plants
fruit crop plants
green crop canopy
plant seedlings
plants in a greenhouse
plants in an orchard
```

### Disease Inspection Context

These prompts are useful for adapter and notebook surfaces where the expected image is a disease-inspection crop.

```text
a close-up of a crop leaf
a close-up of plant disease symptoms
a plant disease inspection image
a diseased crop plant
a healthy crop plant for disease inspection
a leaf with spots
a leaf with lesions
a fruit with plant disease symptoms
a plant part held for inspection
```

### Supported Crop Names

At runtime, append configured crop names from `router.crop_mapping` and taxonomy:

```text
a wheat plant
a maize plant
a barley plant
a sugar beet plant
a sunflower plant
a cotton plant
a tomato plant
a tomato leaf
a tomato fruit
a potato plant
a potato leaf
a grape plant
a strawberry plant
a apricot plant
a hazelnut plant
a apple plant
```

Do not hard-code only these examples. Generate the crop prompts from the repo taxonomy/config so future crops inherit the guard.

## Negative Prompt Groups

The negative side should represent broad non-plant content, not only the examples that triggered the request. Aggregate these into `non_plant_score`.

### Animals And People

```text
an animal
a dog
a cat
a bird
livestock
a person
a human hand without a plant
a face
an insect without a plant
```

Note: insects on a leaf should not be rejected only because an insect score is present. Reject only when positive plant evidence is weak or the negative margin dominates.

### Vehicles And Farm Machinery

```text
a vehicle
a tractor
a truck
a car
a motorcycle
farm machinery
agricultural equipment
a harvesting machine
a plow
a trailer
```

### Buildings, Indoor Scenes, And Infrastructure

```text
a building
a house
a room
an indoor scene
a wall
a road
a street
a fence
a greenhouse structure without visible plants
farm infrastructure without visible plants
```

### Soil, Field, And Background Without Plant Focus

These are important because `field` is ambiguous. A field with visible crop plants should pass; bare soil or distant landscape should reject.

```text
bare soil
a field without visible plants
a distant field landscape
dry ground
mud
rocks
mulch
background scenery
sky and landscape
an empty farm field
```

### Tools, Objects, And Household Items

```text
a tool
a gardening tool
a machine part
a plastic container
a bottle
a phone
a computer screen
a document
a label
a table
a plate
a bag
```

### Food And Harvested Non-Inspection Objects

This group is deliberately weak because some crop-fruit adapters may inspect harvested fruit. Use it as negative support, not a standalone reject.

```text
cooked food
prepared meal
processed food
a grocery item
a fruit on a plate
a vegetable on a table
packaged produce
```

For fruit adapters, down-weight this group or ignore it when configured part is `fruit`.

### Abstract, Graphics, And Bad Inputs

```text
a drawing
a diagram
a screenshot
a chart
text on a page
a logo
a blurry non-plant image
a blank image
a corrupted image
```

## Ambiguity Handling

Do not reject just because a negative prompt has a non-zero score. Reject only when plant evidence is low or negative evidence clearly dominates.

Examples that should normally pass:

- close-up leaf with a hand in the background
- tomato fruit held in a hand
- crop rows with visible leaves
- plant pot on a table
- greenhouse image with clear plants
- harvested fruit when the selected adapter part is `fruit`

Examples that should normally reject:

- dog, cat, livestock, or person without visible plant subject
- tractor, truck, road, building, tool, or screen without visible plant subject
- empty field, bare soil, sky, or distant landscape with no inspectable plant
- label/document/screenshot instead of a real plant image

## Scoring Recommendation

Use the same scoring primitive already available in `src/router/clip_runtime.py`:

- `clip_score_labels_ensemble` for prompt groups when a loaded router/BioCLIP runtime exists.
- For direct adapter smoke tests, create a lightweight guard helper that loads/reuses BioCLIP through the same router loading path where possible.

Recommended aggregation:

1. Score prompts in each group.
2. Use the maximum or top-k mean inside each group.
3. Compute:

```text
plant_score = max(
  generic_plant_score,
  plant_part_score,
  crop_context_score,
  disease_inspection_score,
  supported_crop_score
)

non_plant_score = max(
  animal_people_score,
  vehicle_machinery_score,
  building_scene_score,
  bare_field_background_score,
  tools_objects_score,
  abstract_bad_input_score
)
```

4. Apply part-aware down-weighting:

```text
if requested_part == "fruit":
    food_harvested_score *= 0.4
else:
    food_harvested_score *= 0.8
```

5. Return a structured `input_guard` payload with all group scores under `debug_scores` when diagnostics are enabled.

## Integration Notes

Router-driven inference:

- run the guard before adapter loading
- preserve existing `unknown_crop` and `router_uncertain` statuses
- if guard rejects, return `non_plant_rejected` and skip adapter inference
- run the guard even when `trust_crop_hint=True`

Direct adapter notebooks:

- run the same guard before `adapter.predict_with_ood(...)`
- show the reject as a guard decision, not as a disease prediction
- include `plant_score`, `non_plant_score`, `margin`, and `reason` in the UI and raw JSON
- robust smoke mode should run the guard per view and warn on guard disagreement

## Non-Goals

- Do not add adapter OOD calibration.
- Do not serialize a DINO feature bank into adapter metadata for this guard.
- Do not claim the guard proves disease OOD readiness.
- Do not hard-code the user's examples as the only rejected categories.
