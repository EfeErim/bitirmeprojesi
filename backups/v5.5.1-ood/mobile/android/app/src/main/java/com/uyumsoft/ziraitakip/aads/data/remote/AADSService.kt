package com.uyumsoft.ziraitakip.aads.data.remote

import com.uyumsoft.ziraitakip.aads.data.remote.request.DiagnosisRequest
import com.uyumsoft.ziraitakip.aads.data.remote.response.DiagnosisResponse
import retrofit2.http.*

interface AADSService {

    @POST("v1/diagnose")
    suspend fun diagnose(@Body request: DiagnosisRequest): DiagnosisResponse

    @GET("v1/crops")
    suspend fun getCrops(): Map<String, List<String>>

    @GET("v1/adapters/{crop}/status")
    suspend fun getAdapterStatus(@Path("crop") crop: String): Map<String, Any>

    @POST("v1/feedback/expert-label")
    suspend fun submitExpertLabel(@Body feedback: Map<String, Any>): Map<String, Any>

    @GET("v1/system/info")
    suspend fun getSystemInfo(): Map<String, Any>
}