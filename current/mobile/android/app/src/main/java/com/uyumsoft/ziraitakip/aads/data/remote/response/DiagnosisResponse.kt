package com.uyumsoft.ziraitakip.aads.data.remote.response

data class DiagnosisResponse(
    val status: String,
    val request_id: String? = null,
    val timestamp: String? = null,
    val crop: CropInfo? = null,
    val disease: DiseaseInfo? = null,
    val ood_analysis: OODAnalysis? = null,
    val recommendations: Recommendations? = null,
    val follow_up: FollowUp? = null,
    val message: String? = null
)

data class CropInfo(
    val predicted: String,
    val confidence: Float,
    val from_hint: Boolean? = null
)

data class DiseaseInfo(
    val class_index: Int? = null,
    val name: String? = null,
    val confidence: Float,
    val description: String? = null
)

data class OODAnalysis(
    val is_ood: Boolean,
    val ood_type: String? = null,
    val mahalanobis_distance: Float? = null,
    val threshold: Float? = null,
    val ood_score: Float? = null,
    val nearest_class: String? = null,
    val nearest_distance: Float? = null,
    val confidence: Float? = null
)

data class Recommendations(
    val immediate_actions: List<String>? = null,
    val prevention: List<String>? = null,
    val expert_consultation: Boolean? = null,
    val message: String? = null
)

data class FollowUp(
    val sample_stored: Boolean? = null,
    val sample_id: String? = null,
    val estimated_label_time: String? = null,
    val notification_enabled: Boolean? = null
)