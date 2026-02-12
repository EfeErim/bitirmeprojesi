package com.uyumsoft.ziraitakip.aads.data.remote.request

data class DiagnosisRequest(
    val image: String,  // Base64 encoded image
    val crop_hint: String? = null,
    val location: Location? = null,
    val metadata: Metadata? = null
)

data class Location(
    val latitude: Double,
    val longitude: Double,
    val accuracy_meters: Float? = null
)

data class Metadata(
    val capture_timestamp: String? = null,
    val device_model: String? = null,
    val os_version: String? = null
)