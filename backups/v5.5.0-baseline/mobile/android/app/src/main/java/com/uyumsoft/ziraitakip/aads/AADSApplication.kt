package com.uyumsoft.ziraitakip.aads

import android.app.Application
import androidx.room.Room
import com.uyumsoft.ziraitakip.aads.data.local.AADSDatabase
import com.uyumsoft.ziraitakip.aads.data.remote.AADSService
import com.uyumsoft.ziraitakip.aads.data.repository.DiagnosisRepository
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

class AADSApplication : Application() {

    companion object {
        lateinit var instance: AADSApplication
            private set
    }

    private lateinit var database: AADSDatabase
    private lateinit var aadsService: AADSService
    private lateinit var diagnosisRepository: DiagnosisRepository

    override fun onCreate() {
        super.onCreate()
        instance = this

        // Initialize database
        database = Room.databaseBuilder(
            applicationContext,
            AADSDatabase::class.java,
            "aads_database"
        ).build()

        // Initialize Retrofit
        val loggingInterceptor = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        }

        val okHttpClient = OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build()

        val retrofit = Retrofit.Builder()
            .baseUrl("https://api.uyumsoft.com/") // Configure for production
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        aadsService = retrofit.create(AADSService::class.java)

        // Initialize repository
        diagnosisRepository = DiagnosisRepository(
            aadsService,
            database.diagnosisDao()
        )
    }

    fun getDatabase(): AADSDatabase = database
    fun getAADSService(): AADSService = aadsService
    fun getDiagnosisRepository(): DiagnosisRepository = diagnosisRepository
}