package com.example.qnn_litertlm_gemma

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.URL

/**
 * Utility class for downloading LiteRT-LM models
 */
class ModelDownloader(private val context: Context) {
    
    companion object {
        private const val TAG = "ModelDownloader"
        
        // Gemma3n model URL (you may need to update this with the actual HuggingFace URL)
        private const val GEMMA3N_MODEL_URL = "https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/resolve/main/gemma-3n-E2B-it-int4.litertlm"
        private const val MODEL_FILENAME = "gemma3n.litertlm"
    }
    
    /**
     * Download model with progress reporting
     * @return Flow emitting download progress (0-100)
     */
    fun downloadModel(modelUrl: String = GEMMA3N_MODEL_URL): Flow<DownloadProgress> = flow {
        try {
            emit(DownloadProgress.Started)
            
            val modelFile = File(context.filesDir, MODEL_FILENAME)
            
            // Check if already exists
            if (modelFile.exists()) {
                Log.d(TAG, "Model already exists at ${modelFile.absolutePath}")
                emit(DownloadProgress.Complete(modelFile.absolutePath))
                return@flow
            }
            
            Log.d(TAG, "Downloading model from $modelUrl")
            
            val connection = URL(modelUrl).openConnection()
            connection.connect()
            
            val fileLength = connection.contentLength
            
            connection.getInputStream().use { input ->
                FileOutputStream(modelFile).use { output ->
                    val buffer = ByteArray(4096)
                    var total: Long = 0
                    var count: Int
                    
                    while (input.read(buffer).also { count = it } != -1) {
                        total += count
                        output.write(buffer, 0, count)
                        
                        // Emit progress
                        if (fileLength > 0) {
                            val progress = (total * 100 / fileLength).toInt()
                            emit(DownloadProgress.Progress(progress, total, fileLength.toLong()))
                        }
                    }
                }
            }
            
            Log.d(TAG, "Model downloaded successfully to ${modelFile.absolutePath}")
            emit(DownloadProgress.Complete(modelFile.absolutePath))
            
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading model: ${e.message}", e)
            emit(DownloadProgress.Error(e.message ?: "Unknown error"))
        }
    }.flowOn(Dispatchers.IO)
    
    /**
     * Get the local model path
     */
    fun getModelPath(): String {
        return File(context.filesDir, MODEL_FILENAME).absolutePath
    }
    
    /**
     * Check if model is downloaded
     */
    fun isModelDownloaded(): Boolean {
        return File(getModelPath()).exists()
    }
}

/**
 * Sealed class representing download progress states
 */
sealed class DownloadProgress {
    object Started : DownloadProgress()
    data class Progress(val percentage: Int, val downloaded: Long, val total: Long) : DownloadProgress()
    data class Complete(val filePath: String) : DownloadProgress()
    data class Error(val message: String) : DownloadProgress()
}
